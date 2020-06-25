# Copyright 2016-2020 The Van Valen Lab at the California Institute of
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
"""ImageFileConsumer class for consuming image segmentation jobs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import timeit

import numpy as np

from redis_consumer.consumers import TensorFlowServingConsumer
from redis_consumer import utils
from redis_consumer import settings

from redis_consumer import processing


class MibiConsumer(TensorFlowServingConsumer):
    """Consumes image files and uploads the results"""

    def is_valid_hash(self, redis_hash):
        if redis_hash is None:
            return False

        fname = str(self.redis.hget(redis_hash, 'input_file_name'))
        return not fname.lower().endswith('.zip')

    def _consume(self, redis_hash):
        start = timeit.default_timer()
        hvals = self.redis.hgetall(redis_hash)
        # hold on to the redis hash/values for logging purposes
        self._redis_hash = redis_hash
        self._redis_values = hvals

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

        # Get model_name and version
        model_name, model_version = settings.MIBI_MODEL.split(':')
        self.logger.debug('Model using is: %s', model_name)

        _ = timeit.default_timer()

        # Load input image
        with utils.get_tempdir() as tempdir:
            fname = self.storage.download(hvals.get('input_file_name'), tempdir)
            image = utils.get_image(fname)

        # Pre-process data before sending to the model
        self.update_key(redis_hash, {
            'status': 'pre-processing',
            'download_time': timeit.default_timer() - _,
        })

        # Calculate scale of image and rescale
        scale = hvals.get('scale', '')
        if not scale:
            # Detect scale of image (Default to 1, implement SCALE_DETECT here)
            self.logger.debug('Scale was not given. Defaults to 1')
            scale = 1
        else:
            # Scale was set by user
            self.logger.debug('Image scale was defined as: %s', scale)
        scale = float(scale)
        self.logger.debug('Image scale is: %s', scale)

        # Rescale each channel of the image
        self.logger.debug('Image shape before scaling is: %s', image.shape)
        images = []
        for channel in range(image.shape[0]):
            images.append(utils.rescale(image[channel, ...], scale))
        image = np.concatenate(images, -1)
        self.logger.debug('Image shape after scaling is: %s', image.shape)

        # Preprocess image
        if image.ndim < 4:
            image = np.expand_dims(image, 0)
        image = processing.phase_preprocess(image)
        image = np.squeeze(image)
        self.logger.debug('Shape after phase_preprocess is: %s', image.shape)

        # Send data to the model
        self.update_key(redis_hash, {'status': 'predicting'})
        image = self.predict(image, model_name, model_version)

        print('image type is: ', type(image))

        # Post-process model results
        self.update_key(redis_hash, {'status': 'post-processing'})
        if isinstance(image, list):
            if len(image) == 4:
                image = np.squeeze(processing.deep_watershed_mibi(image))
            else:
                self.logger.warning('Output length was %s, should have been 4')
                image = np.asarray(image)
        else:
            image = image
            self.logger.warning('Output was not in the form of a list')

        self.logger.debug('Shape after deep_watershed_mibi is: %s', image.shape)

        # Save the post-processed results to a file
        _ = timeit.default_timer()
        self.update_key(redis_hash, {'status': 'saving-results'})

        with utils.get_tempdir() as tempdir:
            # Save each result channel as an image file
            save_name = hvals.get('original_name', fname)
            subdir = os.path.dirname(save_name.replace(tempdir, ''))
            name = os.path.splitext(os.path.basename(save_name))[0]

            # Rescale image to original size before sending back to user
            if isinstance(image, list):
                outpaths = []
                for i in image:
                    outpaths.extend(utils.save_numpy_array(
                        utils.rescale(i, 1 / scale), name=name,
                        subdir=subdir, output_dir=tempdir))
            else:
                outpaths = utils.save_numpy_array(
                    utils.rescale(image, 1 / scale), name=name,
                    subdir=subdir, output_dir=tempdir)

            # Save each prediction image as zip file
            zip_file = utils.zip_files(outpaths, tempdir)

            # Upload the zip file to cloud storage bucket
            cleaned = zip_file.replace(tempdir, '')
            subdir = os.path.dirname(settings._strip(cleaned))
            subdir = subdir if subdir else None
            dest, output_url = self.storage.upload(zip_file, subdir=subdir)

        # Update redis with the final results
        t = timeit.default_timer() - start
        self.update_key(redis_hash, {
            'status': self.final_status,
            'output_url': output_url,
            'upload_time': timeit.default_timer() - _,
            'output_file_name': dest,
            'total_jobs': 1,
            'total_time': t,
            'finished_at': self.get_current_timestamp()
        })
        return self.final_status
