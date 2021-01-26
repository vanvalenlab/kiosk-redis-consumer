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
"""MultiplexConsumer class for consuming multiplex segmentation jobs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import timeit

import numpy as np

from deepcell.applications import ScaleDetection, MultiplexSegmentation

from redis_consumer.consumers import TensorFlowServingConsumer
from redis_consumer import settings


class MultiplexConsumer(TensorFlowServingConsumer):
    """Consumes image files and uploads the results"""

    def detect_scale(self, image):
        """Send the image to the SCALE_DETECT_MODEL to detect the relative
        scale difference from the image to the model's training data.

        Args:
            image (numpy.array): The image data.

        Returns:
            scale (float): The detected scale, used to rescale data.
        """
        start = timeit.default_timer()

        app = self.get_grpc_app(settings.SCALE_DETECT_MODEL, ScaleDetection)

        if not settings.SCALE_DETECT_ENABLED:
            self.logger.debug('Scale detection disabled.')
            return 1

        # TODO: What to do with multi-channel data?
        # detected_scale = app.predict(image[..., 0])
        detected_scale = 1

        self.logger.debug('Scale %s detected in %s seconds',
                          detected_scale, timeit.default_timer() - start)

        return detected_scale

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

        # Get model_name and version
        model_name, model_version = settings.MULTIPLEX_MODEL.split(':')

        _ = timeit.default_timer()

        # Load input image
        fname = hvals.get('input_file_name')
        image = self.download_image(fname)

        # squeeze extra dimension that is added by get_image
        image = np.squeeze(image)

        # Pre-process data before sending to the model
        self.update_key(redis_hash, {
            'status': 'pre-processing',
            'download_time': timeit.default_timer() - _,
        })

        # TODO: implement detect_scale here for multiplex model
        scale = hvals.get('scale', '')
        scale = self.get_image_scale(scale, image, redis_hash)

        image = np.expand_dims(image, axis=0)  # add in the batch dim

        # Validate input image
        image = self.validate_model_input(image, model_name, model_version)

        # Send data to the model
        app = self.get_grpc_app(settings.MULTIPLEX_MODEL,
                                MultiplexSegmentation)

        results = app.predict(image, image_mpp=scale * app.model_mpp)

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
