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
"""SegmentationConsumer class for consuming image segmentation jobs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import timeit

import numpy as np

from deepcell.applications import LabelDetection
from deepcell_toolbox.processing import normalize

from redis_consumer.consumers import TensorFlowServingConsumer
from redis_consumer import settings


class SegmentationConsumer(TensorFlowServingConsumer):
    """Consumes image files and uploads the results"""

    def detect_label(self, image):
        """Send the image to the LABEL_DETECT_MODEL to detect the type of image
        data. The model output is mapped with settings.MODEL_CHOICES.

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
        """Calculate label of image."""
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
        image = np.expand_dims(image, axis=0)  # add a batch dimension

        # Validate input image
        if hvals.get('channels'):
            channels = [int(c) for c in hvals.get('channels').split(',')]
        else:
            channels = None

        # Pre-process data before sending to the model
        self.update_key(redis_hash, {
            'status': 'pre-processing',
            'download_time': timeit.default_timer() - _,
        })

        # Calculate scale of image and rescale
        scale = hvals.get('scale', '')
        scale = self.get_image_scale(scale, image, redis_hash)

        label = hvals.get('label', '')
        label = self.get_image_label(label, image, redis_hash)

        # Grap appropriate model and application class
        model = settings.MODEL_CHOICES[label]
        app_cls = settings.APPLICATION_CHOICES[label]

        model_name, model_version = model.split(':')

        # Validate input image
        image = self.validate_model_input(image, model_name, model_version,
                                          channels=channels)

        # Send data to the model
        self.update_key(redis_hash, {'status': 'predicting'})

        app = self.get_grpc_app(model, app_cls)

        results = app.predict(image, batch_size=None,
                              image_mpp=scale * app.model_mpp)

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
