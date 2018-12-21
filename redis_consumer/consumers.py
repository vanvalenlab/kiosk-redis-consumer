# Copyright 2016-2018 The Van Valen Lab at the California Institute of
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
"""Classes to consume events in redis"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from timeit import default_timer

import os
import json
import logging

import grpc
import numpy as np

from redis_consumer.grpc_client import GrpcClient

from redis_consumer import utils
from redis_consumer import settings


class Consumer(object):  # pylint: disable=useless-object-inheritance
    """Base class for all redis event consumer classes"""

    def __init__(self, redis_client, storage_client, final_status='done'):
        self.output_dir = settings.OUTPUT_DIR
        self.redis = redis_client
        self.storage = storage_client
        self.final_status = final_status
        self.logger = logging.getLogger(str(self.__class__.__name__))

    def iter_redis_hashes(self, status='new', prefix='predict'):
        """Iterate over hash values in redis
        and yield each with the given status value.
        # Returns: Iterator of all hashes with a valid status
        """
        for key in self.redis.keys():
            # Check if the key is a hash
            if self.redis.type(key) == 'hash':
                # Check if necessary to filter based on prefix
                if prefix is not None:
                    # Wrong prefix, skip it.
                    if not key.startswith(str(prefix).lower()):
                        continue

                # if status is given, only yield hashes with that status
                if status is not None:
                    if self.redis.hget(key, 'status') == str(status):
                        yield key
                else:  # no need to check the status
                    yield key

    def _process(self, image, key, process_type):
        """Apply each processing function to each image in images
        # Arguments:
            images: iterable of image data
            key: function to apply to images
            process_type: pre or post processing
        # Returns:
            list of processed image data
        """
        if not key:
            return image

        start = default_timer()
        process_type = str(process_type).lower()
        processing_function = utils.get_processing_function(process_type, key)
        self.logger.debug('Starting %s %s-processing image of shape %s',
                          key, process_type, image.shape)
        try:
            results = processing_function(image)
            self.logger.debug('Finished %s %s-processing image in %ss',
                              key, process_type, default_timer() - start)
            return results
        except Exception as err:
            self.logger.error('Encountered %s during %s %s-processing: %s',
                              type(err).__name__, key, process_type, err)
            raise err

    def preprocess(self, image, key):
        """Wrapper for _process_image but can only call with type="pre"
        # Arguments:
            image: numpy array of image data
            key: function to apply to image
        # Returns:
            pre-processed image data
        """
        return self._process(image, key, 'pre')

    def postprocess(self, image, key):
        """Wrapper for _process_image but can only call with type="post"
        # Arguments:
            image: numpy array of image data
            key: function to apply to image
        # Returns:
            post-processed image data
        """
        return self._process(image, key, 'post')

    def _handle_error(self, err, redis_hash):
        # Update redis with failed status
        self.redis.hmset(redis_hash, {
            'reason': '{}'.format(err),
            'status': 'failed'
        })
        self.logger.error('Failed to process redis key %s. %s: %s',
                          redis_hash, type(err).__name__, err)

    def _consume(self, redis_hash):
        raise NotImplementedError

    def consume(self, status=None, prefix=None):
        """Consume all redis events every `interval` seconds
        # Arguments:
            interval: waits this many seconds between consume calls
        # Returns:
            nothing: this is the consumer main process
        """
        try:
            # process each unprocessed hash
            for redis_hash in self.iter_redis_hashes(status, prefix):
                start = default_timer()
                self._consume(redis_hash)
                self.logger.debug('Consumed key %s in %ss',
                                  redis_hash, default_timer() - start)
        except Exception as err:  # pylint: disable=broad-except
            self.logger.error(err)


class PredictionConsumer(Consumer):
    """Consumer to send image data to tf-serving and upload the results"""

    def process_big_image(self,
                          cuts,
                          img,
                          field,
                          model_name,
                          model_version):
        """Slice big image into smaller images for prediction,
        then stitches all the smaller images back together
        # Arguments:
            cuts: number of cuts in x and y to slice smaller images
            img: image data as numpy array
            field: receptive field size of model, changes padding sizes
            model_name: hosted model to send image data
            model_version: model version to query
        # Returns:
            tf_results: single numpy array of predictions on big input image
        """
        cuts = int(cuts)
        winx, winy = (field - 1) // 2, (field - 1) // 2

        def iter_cuts(img, cuts, field):
            padded_img = utils.pad_image(img, field)
            crop_x = img.shape[img.ndim - 3] // cuts
            crop_y = img.shape[img.ndim - 2] // cuts
            for i in range(cuts):
                for j in range(cuts):
                    a, b = i * crop_x, (i + 1) * crop_x
                    c, d = j * crop_y, (j + 1) * crop_y
                    data = padded_img[..., a:b + 2 * winx, c:d + 2 * winy, :]
                    coord = (a, b, c, d)
                    yield data, coord

        slcs, coords = zip(*iter_cuts(img, cuts, field))
        reqs = (self.grpc_image(s, model_name, model_version) for s in slcs)

        tf_results = None
        for resp, (a, b, c, d) in zip(reqs, coords):
            # resp = await asyncio.ensure_future(req)
            if tf_results is None:
                tf_results = np.zeros(list(img.shape)[:-1] + [resp.shape[-1]])
                self.logger.debug('Initialized output tensor of shape %s',
                                  tf_results.shape)

            tf_results[..., a:b, c:d, :] = resp[..., winx:-winx, winy:-winy, :]

        return tf_results

    def grpc_image(self, img, model_name, model_version, timeout=15):
        start = default_timer()
        self.logger.debug('Segmenting image of shape %s with model %s:%s',
                          img.shape, model_name, model_version)
        try:
            hostname = '{}:{}'.format(settings.TF_HOST, settings.TF_PORT)
            req_data = [{'in_tensor_name': settings.TF_TENSOR_NAME,
                         'in_tensor_dtype': settings.TF_TENSOR_DTYPE,
                         'data': np.expand_dims(img, axis=0)}]

            client = GrpcClient(hostname, model_name, int(model_version))
            prediction = client.predict(req_data, request_timeout=timeout)
            self.logger.debug('Segmented image with model %s:%s in %ss',
                              model_name, model_version,
                              default_timer() - start)
            return prediction['prediction']
        except grpc.RpcError as err:
            retry_statuses = {
                grpc.StatusCode.DEADLINE_EXCEEDED,
                grpc.StatusCode.UNAVAILABLE
            }
            code = err.code() if hasattr(err, 'code') else None
            if code in retry_statuses:
                self.logger.warning(err.details())
                self.logger.warning('Encountered %s during tf-serving request '
                                    'to model %s:%s: %s', type(err).__name__,
                                    model_name, model_version, err)
                return self.grpc_image(img, model_name, model_version, timeout)
            raise err
        except Exception as err:
            self.logger.error('Encountered %s during tf-serving request to '
                              'model %s:%s: %s', type(err).__name__,
                              model_name, model_version, err)
            raise err

    def _consume(self, redis_hash):
        # TODO: investigate parallel requests
        hvals = self.redis.hgetall(redis_hash)
        self.logger.debug('Found hash to process "%s": %s',
                          redis_hash, json.dumps(hvals, indent=4))

        self.redis.hset(redis_hash, 'status', 'processing')

        model_name = hvals.get('model_name')
        model_version = hvals.get('model_version')
        cuts = hvals.get('cuts', '0')
        field = hvals.get('field_size', '61')

        try:
            with utils.get_tempdir() as tempdir:
                fname = self.storage.download(hvals.get('file_name'), tempdir)
                image_files = utils.get_image_files_from_dir(fname, tempdir)

                all_output = []
                for i, imfile in enumerate(image_files):
                    start = default_timer()
                    image = utils.get_image(imfile)

                    pre = None
                    for f in hvals.get('preprocess_function', '').split(','):
                        x = pre if pre else image
                        pre = self.preprocess(x, f)

                    if cuts.isdigit() and int(cuts) > 0:
                        prediction = self.process_big_image(
                            cuts, pre, field, model_name, model_version)
                    else:
                        prediction = self.grpc_image(
                            pre, model_name, model_version, timeout=30)

                    post = None
                    for f in hvals.get('postprocess_function', '').split(','):
                        x = post if post else prediction
                        post = self.postprocess(x, f)

                    # Save each result channel as an image file
                    subdir = os.path.dirname(imfile.replace(tempdir, ''))
                    name = os.path.splitext(os.path.basename(imfile))[0]

                    _out_paths = utils.save_numpy_array(
                        post, name=name, subdir=subdir, output_dir=tempdir)

                    all_output.extend(_out_paths)
                    self.logger.info('Saved data for image %s in %ss',
                                     i, default_timer() - start)

                # Save each prediction image as zip file
                zip_file = utils.zip_files(all_output, tempdir)

                # Upload the zip file to cloud storage bucket
                uploaded_file_path = self.storage.upload(zip_file)

            output_url = self.storage.get_public_url(uploaded_file_path)
            self.logger.debug('Uploaded output to: "%s"', output_url)

            # Update redis with the results
            self.redis.hmset(redis_hash, {
                'output_url': output_url,
                'status': self.final_status
            })
            self.logger.debug('Updated status to %s', self.final_status)

        except Exception as err:  # pylint: disable=broad-except
            self._handle_error(err, redis_hash)
