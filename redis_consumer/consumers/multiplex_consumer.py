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

import os
import timeit

import numpy as np

from redis_consumer.consumers import ImageFileConsumer
from redis_consumer import utils
from redis_consumer import settings


class MultiplexConsumer(ImageFileConsumer):
    """Consumes image files and uploads the results"""

    def _consume(self, redis_hash):
        start = timeit.default_timer()
        self._redis_hash = redis_hash  # workaround for logging.
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
        with utils.get_tempdir() as tempdir:
            fname = self.storage.download(hvals.get('input_file_name'), tempdir)
            # TODO: tiffs expand the last axis, is that a problem here?
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
            # TODO: implement SCALE_DETECT here for multiplex model
            # scale = self.detect_scale(image)
            # self.logger.debug('Image scale detected: %s', scale)
            # self.update_key(redis_hash, {'scale': scale})
            self.logger.debug('Scale was not given. Defaults to 1')
            scale = 1
        else:
            scale = float(scale)
            self.logger.debug('Image scale already calculated %s', scale)

        # squeeze out extra channel dimension added by 'get_image'
        image = np.squeeze(image)

        # if for some reason a one-channel image was passed, add the channel back if __name__ == '__main__':
        if image.ndim == 2:
            image = np.expand_dims(image, -1)

        # Rescale each channel of the image
        image = utils.rescale(image, scale)
        image = np.expand_dims(image, axis=0)  # add in the batch dim

        # Preprocess image
        image = self.preprocess(image, ['histogram_normalization'])

        # Send data to the model
        self.update_key(redis_hash, {'status': 'predicting'})
        image = self.predict(image, model_name, model_version)

        # Post-process model results
        self.update_key(redis_hash, {'status': 'post-processing'})
        image = self.postprocess(image, ['multiplex'])

        # Save the post-processed results to a file
        _ = timeit.default_timer()
        self.update_key(redis_hash, {'status': 'saving-results'})

        save_name = hvals.get('original_name', fname)
        dest, output_url = self.save_output(image, redis_hash, save_name, scale)

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
