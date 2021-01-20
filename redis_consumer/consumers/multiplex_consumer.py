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

from redis_consumer.consumers import TensorFlowServingConsumer
from redis_consumer import utils
from redis_consumer import settings
from redis_consumer import processing


class MultiplexConsumer(TensorFlowServingConsumer):
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

        # squeeze extra dimension that is added by get_image
        image = np.squeeze(image)

        # validate correct shape of image
        if len(image.shape) > 3:
            raise ValueError('Invalid image shape. An image of shape {} was supplied, but the '
                             'multiplex model expects of images of shape'
                             '[height, widths, 2]'.format(image.shape))
        elif len(image.shape) < 3:
            # TODO: Once we can pass warning messages to user, we can treat this as nuclear image
            raise ValueError('Invalid image shape. An image of shape {} was supplied, but the '
                             'multiplex model expects images of shape'
                             '[height, width, 2]'.format(image.shape))
        else:
            if image.shape[0] == 2:
                image = np.rollaxis(image, 0, 3)
            elif image.shape[2] == 2:
                pass
            else:
                raise ValueError('Invalid image shape. An image of shape {} was supplied, '
                                 'but the multiplex model expects images of shape'
                                 '[height, widths, 2]'.format(image.shape))

        # Pre-process data before sending to the model
        self.update_key(redis_hash, {
            'status': 'pre-processing',
            'download_time': timeit.default_timer() - _,
        })

        # TODO: implement detect_scale here for multiplex model
        scale = hvals.get('scale', '')
        scale = self.get_image_scale(scale, image, redis_hash)

        original_shape = image.shape

        # Rescale each channel of the image
        image = utils.rescale(image, scale)
        image = np.expand_dims(image, axis=0)  # add in the batch dim

        # Preprocess image
        image = self.preprocess(image, ['multiplex_preprocess'])

        # Send data to the model
        self.update_key(redis_hash, {'status': 'predicting'})
        image = self.predict(image, model_name, model_version)

        # Post-process model results
        self.update_key(redis_hash, {'status': 'post-processing'})
        image = processing.format_output_multiplex(image)
        image = self.postprocess(image, ['multiplex_postprocess_consumer'])

        # Save the post-processed results to a file
        _ = timeit.default_timer()
        self.update_key(redis_hash, {'status': 'saving-results'})

        save_name = hvals.get('original_name', fname)
        dest, output_url = self.save_output(
            image, redis_hash, save_name, original_shape[:-1])

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
