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


class ImageFileConsumer(TensorFlowServingConsumer):
    """Consumes image files and uploads the results"""

    def is_valid_hash(self, redis_hash):
        if redis_hash is None:
            return False

        fname = str(self.redis.hget(redis_hash, 'input_file_name'))
        return not fname.lower().endswith('.zip')

    def detect_scale(self, image):
        """Send the image to the SCALE_DETECT_MODEL to detect the relative
        scale difference from the image to the model's training data.

        Args:
            image (numpy.array): The image data.

        Returns:
            scale (float): The detected scale, used to rescale data.
        """
        start = timeit.default_timer()

        if not settings.SCALE_DETECT_ENABLED:
            self.logger.debug('Scale detection disabled. Scale set to 1.')
            return 1

        model_name, model_version = settings.SCALE_DETECT_MODEL.split(':')

        scales = self.predict(image, model_name, model_version,
                              untile=False)

        detected_scale = np.mean(scales)

        error_rate = .01  # error rate is ~1% for current model.
        if abs(detected_scale - 1) < error_rate:
            detected_scale = 1

        self.logger.debug('Scale %s detected in %s seconds',
                          detected_scale, timeit.default_timer() - start)
        return detected_scale

    def detect_label(self, image):
        """Send the image to the LABEL_DETECT_MODEL to detect the type of image
        data. The model output is mapped with settings.MODEL_CHOICES.

        Args:
            image (numpy.array): The image data.

        Returns:
            label (int): The detected label.
        """
        start = timeit.default_timer()

        if not settings.LABEL_DETECT_ENABLED:
            self.logger.debug('Label detection disabled. Label set to None.')
            return None

        model_name, model_version = settings.LABEL_DETECT_MODEL.split(':')

        labels = self.predict(image, model_name, model_version,
                              untile=False)

        labels = np.array(labels)
        vote = labels.sum(axis=0)
        maj = vote.max()

        detected = np.where(vote == maj)[-1][0]

        self.logger.debug('Label %s detected in %s seconds.',
                          detected, timeit.default_timer() - start)
        return detected

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

        # Overridden with LABEL_DETECT_ENABLED
        model_name = hvals.get('model_name')
        model_version = hvals.get('model_version')

        _ = timeit.default_timer()

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
        scale = self.get_image_scale(scale, image, redis_hash)
        image = utils.rescale(image, scale)

        # Save shape value for postprocessing purposes
        # TODO this is a big janky
        self._rawshape = image.shape
        label = None
        if settings.LABEL_DETECT_ENABLED and model_name and model_version:
            self.logger.warning('Label Detection is enabled, but the model'
                                ' %s:%s was specified in Redis.',
                                model_name, model_version)

        elif settings.LABEL_DETECT_ENABLED:
            # Detect image label type
            label = hvals.get('label', '')
            if not label:
                label = self.detect_label(image)
                self.logger.debug('Image label detected: %s', label)
                self.update_key(redis_hash, {'label': str(label)})
            else:
                label = int(label)
                self.logger.debug('Image label already calculated: %s', label)

            # Grap appropriate model
            model_name, model_version = utils._pick_model(label)

        if settings.LABEL_DETECT_ENABLED and label is not None:
            pre_funcs = utils._pick_preprocess(label).split(',')
        else:
            pre_funcs = hvals.get('preprocess_function', '').split(',')

        image = np.expand_dims(image, axis=0)  # add in the batch dim
        image = self.preprocess(image, pre_funcs)

        # Send data to the model
        self.update_key(redis_hash, {'status': 'predicting'})

        image = self.predict(image, model_name, model_version)

        # Post-process model results
        self.update_key(redis_hash, {'status': 'post-processing'})

        if settings.LABEL_DETECT_ENABLED and label is not None:
            post_funcs = utils._pick_postprocess(label).split(',')
        else:
            post_funcs = hvals.get('postprocess_function', '').split(',')

        image = self.postprocess(image, post_funcs)

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
