# Copyright 2016-2022 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/kiosk-redis-consumer/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""SegmentationConsumer class for consuming image segmentation jobs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import timeit

import numpy as np

from deepcell.applications import LabelDetection

from redis_consumer.consumers import TensorFlowServingConsumer
from redis_consumer import settings
from redis_consumer import utils


class SegmentationConsumer(TensorFlowServingConsumer):
    """Consumes image files and uploads the results"""

    def detect_label(self, image):
        """ DEPRECATED -- Send the image to the LABEL_DETECT_MODEL to
        detect the type of image data. The model output is mapped with
        settings.MODEL_CHOICES.

        Args:
            image (numpy.array): The image data.

        Returns:
            label (int): The detected label.
        """
        start = timeit.default_timer()

        app = self.get_grpc_app(settings.LABEL_DETECT_MODEL, LabelDetection)

        if not settings.LABEL_DETECT_ENABLED:
            self.logger.debug('Label detection disabled. Label set to 0.')
            return 0  # Use NuclearSegmentation as default model

        batch_size = app.model.get_batch_size()
        detected_label = app.predict(image, batch_size=batch_size)

        self.logger.debug('Label %s detected in %s seconds',
                          detected_label, timeit.default_timer() - start)

        return int(detected_label)

    def get_image_label(self, label, image, redis_hash):
        """ DEPRECATED -- Calculate label of image."""
        if not label:
            # Detect scale of image (Default to 1)
            label = self.detect_label(image)
            self.logger.debug('Image label detected: %s.', label)
            self.update_key(redis_hash, {'label': label})
        else:
            label = int(label)
            self.logger.debug('Image label already calculated: %s.', label)
            if label not in settings.APPLICATION_CHOICES:
                raise ValueError('Label type {} is not supported'.format(label))

        return label

    def image_dimensions_to_bxyc(self, dim_order, image):
        """Modifies image dimensions to be BXYC."""

        if len(np.shape(image)) != len(dim_order):
            raise ValueError('Input dimension order was {} but input '
                             'image has shape {}'.format(dim_order, np.shape(image)))

        if dim_order == 'XYB':
            image = np.moveaxis(image, -1, 0)
        elif dim_order == 'CXY':
            image = np.moveaxis(image, 0, -1)
        elif dim_order == 'CXYB':
            image = np.swapaxes(image, 0, -1)

        if 'B' not in dim_order:
            image = np.expand_dims(image, axis=0)
        if 'C' not in dim_order:
            image = np.expand_dims(image, axis=-1)

        return(image)

    def save_output(self, image, save_name):
        """Save output images into a zip file and upload it."""
        with tempfile.TemporaryDirectory() as tempdir:
            # Save each result channel as an image file
            subdir = os.path.dirname(save_name.replace(tempdir, ''))
            name = os.path.splitext(os.path.basename(save_name))[0]

            if not isinstance(image, list):
                image = [image]

            outpaths = []
            for i, img in enumerate(image):
                outpaths.extend(utils.save_numpy_array(
                    img,
                    name=str(name) + '_batch_{}'.format(i),
                    subdir=subdir, output_dir=tempdir))

            # Save each prediction image as zip file
            zip_file = utils.zip_files(outpaths, tempdir)

            # Upload the zip file to cloud storage bucket
            cleaned = zip_file.replace(tempdir, '')
            subdir = os.path.dirname(utils.strip_bucket_path(cleaned))
            subdir = subdir if subdir else None
            dest, output_url = self.storage.upload(zip_file, subdir=subdir)

        return dest, output_url

    def _consume(self, redis_hash):
        start = timeit.default_timer()
        hvals = self.redis.hgetall(redis_hash)

        if hvals.get('status') in self.finished_statuses:
            self.logger.warning('Found completed hash `%s` with status %s.',
                                redis_hash, hvals.get('status'))
            return hvals.get('status')

        self.logger.debug('Found hash to process `%s` with status `%s`.',
                          redis_hash, hvals.get('status'))

        self.update_key(redis_hash, {
            'status': 'started',
            'identity_started': self.name,
        })

        _ = timeit.default_timer()

        # Load input image
        fname = hvals.get('input_file_name')
        image = self.download_image(fname)
        dim_order = hvals.get('dimension_order')

        # Modify image dimensions to be BXYC
        image = self.image_dimensions_to_bxyc(dim_order, image)

        # Pre-process data before sending to the model
        self.update_key(redis_hash, {
            'status': 'pre-processing',
            'download_time': timeit.default_timer() - _,
        })

        # Calculate scale of image and rescale
        scale = hvals.get('scale', '')
        scale = self.get_image_scale(scale, image, redis_hash)

        # Validate input image
        channels = hvals.get('channels').split(',')  # ex: channels = ['0','1','2']

        results = []
        for i in range(len(channels)):
            if channels[i]:
                slice_image = image[..., int(channels[i])]
                slice_image = np.expand_dims(slice_image, axis=-1)
                # Grap appropriate model and application class
                model = settings.MODEL_CHOICES[i]
                app_cls = settings.APPLICATION_CHOICES[i]

                model_name, model_version = model.split(':')

                # Validate input image
                slice_image = self.validate_model_input(slice_image, model_name, model_version)

                # Send data to the model
                self.update_key(redis_hash, {'status': 'predicting'})

                app = self.get_grpc_app(model, app_cls)
                # with new batching update in deepcell.applications,
                # app.predict() cannot handle a batch_size of None.
                batch_size = min(32, app.model.get_batch_size())  # TODO: raise max batch size
                pred_results = app.predict(slice_image, batch_size=batch_size,
                                           image_mpp=scale * app.model_mpp)

                results.extend(pred_results)

        # Save the post-processed results to a file
        _ = timeit.default_timer()
        self.update_key(redis_hash, {'status': 'saving-results'})

        save_name = hvals.get('original_name', fname)
        dest, output_url = self.save_output(results, save_name)

        # Update redis with the final results
        end = timeit.default_timer()
        self.update_key(redis_hash, {
            'status': self.final_status,
            'output_url': output_url,
            'upload_time': end - _,
            'output_file_name': dest,
            'total_jobs': 1,
            'total_time': end - start,
            'finished_at': self.get_current_timestamp()
        })
        return self.final_status
