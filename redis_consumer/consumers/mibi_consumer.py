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
"""MibiConsumer class for consuming image segmentation jobs."""
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
        # self._redis_hash = redis_hash
        # self._redis_values = hvals

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
            # Detect scale of image (Default to 1)
            # TODO implement SCALE_DETECT here for mibi model
            # scale = self.detect_scale(image[..., 0])
            # self.logger.debug('Image scale detected: %s', scale)
            # self.update_key(redis_hash, {'scale': scale})
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
        for channel in range(image.shape[2]):
            images.append(utils.rescale(image[..., channel, :], scale))
        image = np.concatenate(images, -1)
        self.logger.debug('Image shape after scaling is: %s', image.shape)

        # Preprocess image
        # Image must be of shape (batch, x, y, channels), but scaling
        # eliminates batch dim, so we recreate it here
        image = np.expand_dims(image, 0)
        image = processing.phase_preprocess(image)
        self.logger.debug('Shape after phase_preprocess is: %s', image.shape)
        image = np.squeeze(image)
        self.logger.debug('Shape after squeeze is: %s', image.shape)

        # Send data to the model
        self.update_key(redis_hash, {'status': 'predicting'})
        image = self.predict(image, model_name, model_version)

        # Post-process model results
        self.update_key(redis_hash, {'status': 'post-processing'})
        image = np.squeeze(processing.deep_watershed_mibi(image))
        self.logger.debug('Shape after deep_watershed_mibi is: %s', image.shape)

        # Save the post-processed results to a file
        _ = timeit.default_timer()
        self.update_key(redis_hash, {'status': 'saving-results'})

        dest, output_url = self.save_and_upload(image, redis_hash, fname, scale)

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
