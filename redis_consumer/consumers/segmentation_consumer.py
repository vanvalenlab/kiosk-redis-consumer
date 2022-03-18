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

import timeit

import numpy as np

from deepcell.applications import LabelDetection

from redis_consumer.consumers import TensorFlowServingConsumer
from redis_consumer import settings


class SegmentationConsumer(TensorFlowServingConsumer):
    """Consumes image files and uploads the results"""

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
        image = np.squeeze(image)
        image = np.expand_dims(image, axis=0)  # add a batch dimension
        if len(np.shape(image)) == 3:
            image = np.expand_dims(image, axis=-1)  # add a channel dimension

        self.logger.debug('Image shape 1: {}'.format(np.shape(image)))
        rank = 4  # (b,x,y,c)
        channel_axis = image.shape[1:].index(min(image.shape[1:])) + 1
        if channel_axis != rank - 1:
            image = np.rollaxis(image, 1, rank)
        self.logger.debug('Image shape 2: {}'.format(np.shape(image)))

        # Pre-process data before sending to the model
        self.update_key(redis_hash, {
            'status': 'pre-processing',
            'download_time': timeit.default_timer() - _,
        })

        # Calculate scale of image and rescale
        scale = hvals.get('scale', '')
        scale = self.get_image_scale(scale, image, redis_hash)

        # Validate input image
        if hvals.get('channels'):
            channels = hvals.get('channels').split(',')
        else:
            channels = None
        self.logger.debug('Channels: {}'.format(channels))

        if channels[0]:
            nuc_image = image[..., int(channels[0])]
            nuc_image = np.expand_dims(nuc_image, axis=-1)
            # Grap appropriate model and application class
            model = settings.MODEL_CHOICES[0]
            app_cls = settings.APPLICATION_CHOICES[0]

            model_name, model_version = model.split(':')

            # detect dimension order and add to redis
            dim_order = self.detect_dimension_order(nuc_image, model_name, model_version)
            self.update_key(redis_hash, {
                'dim_order': ','.join(dim_order)
            })

            # Validate input image
            nuc_image = self.validate_model_input(nuc_image, model_name, model_version)

            # Send data to the model
            self.update_key(redis_hash, {'status': 'predicting'})

            app = self.get_grpc_app(model, app_cls)
            # with new batching update in deepcell.applications,
            # app.predict() cannot handle a batch_size of None.
            batch_size = max(32, app.model.get_batch_size())  # TODO: raise max batch size
            nuc_results = app.predict(nuc_image, batch_size=batch_size,
                                      image_mpp=scale * app.model_mpp)

        if channels[1]:
            cyto_image = image[..., int(channels[1])]
            cyto_image = np.expand_dims(cyto_image, axis=-1)
            # Grap appropriate model and application class
            model = settings.MODEL_CHOICES[2]
            app_cls = settings.APPLICATION_CHOICES[2]

            model_name, model_version = model.split(':')

            # detect dimension order and add to redis
            dim_order = self.detect_dimension_order(cyto_image, model_name, model_version)
            self.update_key(redis_hash, {
                'dim_order': ','.join(dim_order)
            })

            # Validate input image
            cyto_image = self.validate_model_input(cyto_image, model_name, model_version)

            # Send data to the model
            self.update_key(redis_hash, {'status': 'predicting'})

            app = self.get_grpc_app(model, app_cls)
            # with new batching update in deepcell.applications,
            # app.predict() cannot handle a batch_size of None.
            batch_size = max(32, app.model.get_batch_size())  # TODO: raise max batch size
            cyto_results = app.predict(cyto_image, batch_size=batch_size,
                                       image_mpp=scale * app.model_mpp)

        if channels[0] and channels[1]:
            results = np.concatenate((nuc_results, cyto_results), axis=-1)
        elif channels[0]:
            results = nuc_results
        elif channels[1]:
            results = cyto_results

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
