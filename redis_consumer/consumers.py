# Copyright 2016-2019 The Van Valen Lab at the California Institute of
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

import timeit
import uuid

import os
import json
import time
import logging
import zipfile

import grpc
import numpy as np
from redis.exceptions import ConnectionError

from redis_consumer.grpc_clients import PredictClient
from redis_consumer.grpc_clients import ProcessClient
from redis_consumer import utils
from redis_consumer import settings


class Consumer(object):  # pylint: disable=useless-object-inheritance
    """Base class for all redis event consumer classes.

    Args:
        redis_client: Client class to communicate with redis
        storage_client: Client to communicate with cloud storage buckets.
        final_status: Update the status of redis event with this value.
    """

    def __init__(self,
                 redis_client,
                 storage_client,
                 final_status='done',
                 redis_retry_timeout=settings.REDIS_TIMEOUT):
        self.output_dir = settings.OUTPUT_DIR
        self.redis = redis_client
        self.storage = storage_client
        self.final_status = final_status
        self._redis_retry_timeout = redis_retry_timeout
        self.logger = logging.getLogger(str(self.__class__.__name__))
        self.hostname = settings.HOSTNAME

    def iter_redis_hashes(self, status='new', prefix='predict'):
        """Iterate over hash values in redis.
        Yield each with the given status value.

        Returns:
            Iterator of all hashes with a valid status
        """
        match = '%s*' % str(prefix).lower() if prefix is not None else None
        for key in self.scan_iter(match=match):
            # Check if the key is a hash
            if self._redis_type(key) == 'hash':
                # if status is given, only yield hashes with that status
                if status is not None:
                    if self.hget(key, 'status') == str(status):
                        yield key
                else:  # no need to check the status
                    yield key

    def _handle_error(self, err, redis_hash):
        """Update redis with failure information, and log errors.

        Args:
            err: Exception, uncaught error that will be logged.
            redis_hash: string, the hash that will be updated to failure.
        """
        # Update redis with failed status
        ts = time.time() * 1000
        self.hmset(redis_hash, {
            'reason': '{}: {}'.format(type(err).__name__, err),
            'status': 'failed',
            'timestamp_failed': ts,
            'identity_failed': self.hostname,
            'timestamp_last_status_update': ts
        })
        self.logger.error('Failed to process redis key %s due to %s: %s',
                          redis_hash, type(err).__name__, err)

    def _consume(self, redis_hash):
        raise NotImplementedError

    def _redis_type(self, redis_key):
        while True:
            try:
                response = self.redis.type(redis_key)
                break
            except ConnectionError as err:
                self.logger.warning('Encountered %s: %s when calling '
                                    '`TYPE %s`. Retrying in %s seconds.',
                                    type(err).__name__, err, redis_key,
                                    self._redis_retry_timeout)
                time.sleep(self._redis_retry_timeout)
        return response

    def scan_iter(self, match=None):
        while True:
            try:
                start = timeit.default_timer()
                response = self.redis.scan_iter(match=match)
                self.logger.debug('Finished SCAN in %s seconds.',
                                  timeit.default_timer() - start)
                break
            except ConnectionError as err:
                self.logger.warning('Encountered %s: %s when calling '
                                    'SCAN. Retrying in %s seconds.',
                                    type(err).__name__, err,
                                    self._redis_retry_timeout)
                time.sleep(self._redis_retry_timeout)
        return response

    def keys(self):
        while True:
            try:
                start = timeit.default_timer()
                response = self.redis.keys()
                self.logger.debug('KEYS got %s results in %s seconds.',
                                  len(response), timeit.default_timer() - start)
                break
            except ConnectionError as err:
                self.logger.warning('Encountered %s: %s when calling '
                                    'KEYS. Retrying in %s seconds.',
                                    type(err).__name__, err,
                                    self._redis_retry_timeout)
                time.sleep(self._redis_retry_timeout)
        return response

    def hset(self, rhash, key, value):
        while True:
            try:
                response = self.redis.hset(rhash, key, value)
                break
            except ConnectionError as err:
                self.logger.warning('Encountered %s: %s when calling '
                                    '`HSET %s %s %s`. Retrying in %s seconds.',
                                    type(err).__name__, err, rhash, key, value,
                                    self._redis_retry_timeout)
                time.sleep(self._redis_retry_timeout)
        return response

    def hget(self, rhash, key):
        while True:
            try:
                response = self.redis.hget(rhash, key)
                break
            except ConnectionError as err:
                self.logger.warning('Encountered %s: %s when calling '
                                    '`HGET %s %s`. Retrying in %s seconds.',
                                    type(err).__name__, err, rhash, key,
                                    self._redis_retry_timeout)
                time.sleep(self._redis_retry_timeout)
        return response

    def hmset(self, rhash, data):
        while True:
            try:
                response = self.redis.hmset(rhash, data)
                self.logger.debug('Updated hash %s with values: %s.',
                                  rhash, data)
                break
            except ConnectionError as err:
                self.logger.warning('Encountered %s: %s when calling '
                                    '`HMSET %s %s`. Retrying in %s seconds.',
                                    type(err).__name__, err, rhash, data,
                                    self._redis_retry_timeout)
                time.sleep(self._redis_retry_timeout)
        return response

    def hgetall(self, rhash):
        while True:
            try:
                response = self.redis.hgetall(rhash)
                break
            except ConnectionError as err:
                self.logger.warning('Encountered %s: %s when calling '
                                    '`HGETALL %s`. Retrying in %s seconds.',
                                    type(err).__name__, err, rhash,
                                    self._redis_retry_timeout)
                time.sleep(self._redis_retry_timeout)
        return response

    def consume(self, status=None, prefix=None):
        """Consume all redis events every `interval` seconds.

        Args:
            status: string, only consume hashes where `status` == status.
            prefix: string, only consume hashes that start with `prefix`.

        Returns:
            nothing: this is the consumer main process
        """
        # process each unprocessed hash
        for redis_hash in self.iter_redis_hashes(status, prefix):
            try:
                start = timeit.default_timer()
                self._consume(redis_hash)
                hvals = self.hgetall(redis_hash)
                self.logger.debug('Consumed key %s (model %s:%s, '
                                  'preprocessing: %s, postprocessing: %s) '
                                  '(%s retries) in %s seconds.',
                                  redis_hash, hvals.get('model_name'),
                                  hvals.get('model_version'),
                                  hvals.get('preprocess_function'),
                                  hvals.get('postprocess_function'),
                                  0, timeit.default_timer() - start)
            except Exception as err:  # pylint: disable=broad-except
                self._handle_error(err, redis_hash)


class ImageFileConsumer(Consumer):
    """Consumes image files and uploads the results"""

    def iter_redis_hashes(self, status='new', prefix='predict'):
        """Iterate over hash values in redis.
        Only yield hash values for valid image files

        Returns:
            Iterator of all hashes with a valid status
        """
        keys = super(ImageFileConsumer, self).iter_redis_hashes(status, prefix)
        for key in keys:
            fname = str(self.hget(key, 'input_file_name'))
            if not fname.lower().endswith('.zip'):
                yield key

    def _process(self, image, key, process_type, timeout=30, streaming=False):
        """Apply each processing function to image.

        Args:
            image: numpy array of image data
            key: function to apply to image
            process_type: pre or post processing
            timeout: integer. grpc request timeout.
            streaming: boolean. if True, streams data in multiple requests

        Returns:
            list of processed image data
        """
        # Squeeze out batch dimension if unnecessary
        if image.shape[0] == 1:
            image = np.squeeze(image, axis=0)

        if not key:
            return image

        self.logger.debug('Starting %s %s-processing image of shape %s',
                          key, process_type, image.shape)

        retrying = True
        count = 0
        start = timeit.default_timer()
        while retrying:
            try:
                key = str(key).lower()
                process_type = str(process_type).lower()
                hostname = '{}:{}'.format(settings.DP_HOST, settings.DP_PORT)
                client = ProcessClient(hostname, process_type, key)

                if streaming:
                    dtype = 'DT_STRING'
                else:
                    dtype = settings.TF_TENSOR_DTYPE

                req_data = [{'in_tensor_name': settings.TF_TENSOR_NAME,
                             'in_tensor_dtype': dtype,
                             'data': np.expand_dims(image, axis=0)}]

                if streaming:
                    results = client.stream_process(req_data, timeout)
                else:
                    results = client.process(req_data, timeout)

                self.logger.debug('%s-processed key %s (model %s:%s, '
                                  'preprocessing: %s, postprocessing: %s)'
                                  ' (%s retries)  in %s seconds.',
                                  process_type.capitalize(), self._redis_hash,
                                  self._redis_values.get('model_name'),
                                  self._redis_values.get('model_version'),
                                  self._redis_values.get('preprocess_function'),
                                  self._redis_values.get('postprocess_function'),
                                  count, timeit.default_timer() - start)

                results = results['results']
                # Again, squeeze out batch dimension if unnecessary
                if results.shape[0] == 1:
                    results = np.squeeze(results, axis=0)

                retrying = False
                return results
            except grpc.RpcError as err:
                retry_statuses = {
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.UNAVAILABLE
                }
                # pylint: disable=E1101
                if err.code() in retry_statuses:
                    count += 1
                    # write update to Redis
                    processing_retry_time = time.time() * 1000
                    self.hmset(self._redis_hash, {
                        'number_of_processing_retries': count,
                        'status': '{} {}-processing -- RETRY:{} -- {}'.format(
                            key, process_type, count,
                            err.code().name),
                        'timestamp_processing_retry': processing_retry_time,
                        'identity_processing_retry': self.hostname,
                        'timestamp_last_status_update': processing_retry_time
                    })
                    sleeptime = np.random.randint(24, 44)
                    sleeptime = 1 + sleeptime * int(streaming)
                    self.logger.warning('%sException `%s: %s` during %s '
                                        '%s-processing request.  Waiting %s '
                                        'seconds before retrying.',
                                        type(err).__name__, err.code().name,
                                        err.details(), key, process_type,
                                        sleeptime)
                    self.logger.debug('Waiting for %s seconds before retrying',
                                      sleeptime)
                    time.sleep(sleeptime)  # sleep before retry
                    retrying = True  # Unneccessary but explicit
                else:
                    retrying = False
                    raise err
            except Exception as err:
                retrying = False
                self.logger.error('Encountered %s during %s %s-processing: %s',
                                  type(err).__name__, key, process_type, err)
                raise err

    def preprocess(self, image, keys, timeout=30, streaming=False):
        """Wrapper for _process_image but can only call with type="pre".

        Args:
            image: numpy array of image data
            keys: list of function names to apply to the image
            timeout: integer. grpc request timeout.
            streaming: boolean. if True, streams data in multiple requests

        Returns:
            pre-processed image data
        """
        pre = None
        for key in keys:
            x = pre if pre else image
            pre = self._process(x, key, 'pre', timeout, streaming)
        return pre

    def postprocess(self, image, keys, timeout=30, streaming=False):
        """Wrapper for _process_image but can only call with type="post".

        Args:
            image: numpy array of image data
            keys: list of function names to apply to the image
            timeout: integer. grpc request timeout.
            streaming: boolean. if True, streams data in multiple requests

        Returns:
            post-processed image data
        """
        post = None
        for key in keys:
            x = post if post else image
            post = self._process(x, key, 'post', timeout, streaming)
        return post

    def process_big_image(self,
                          cuts,
                          img,
                          field,
                          model_name,
                          model_version):
        """Slice big image into smaller images for prediction,
        then stitches all the smaller images back together.

        Args:
            cuts: number of cuts in x and y to slice smaller images
            img: image data as numpy array
            field: receptive field size of model, changes padding sizes
            model_name: hosted model to send image data
            model_version: model version to query

        Returns:
            tf_results: single numpy array of predictions on big input image
        """
        start = timeit.default_timer()
        cuts = int(cuts)
        field = int(field)
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

        self.logger.debug('Segmented image into shape %s in %s s',
                          tf_results.shape, timeit.default_timer() - start)
        return tf_results

    def grpc_image(self, img, model_name, model_version, timeout=30, backoff=3):
        count = 0
        start = timeit.default_timer()
        self.logger.debug('Segmenting image of shape %s with model %s:%s',
                          img.shape, model_name, model_version)
        retrying = True
        while retrying:
            try:
                floatx = settings.TF_TENSOR_DTYPE
                if 'f16' in model_name:
                    floatx = 'DT_HALF'
                    # TODO: seems like should cast to "half"
                    # but the model rejects the type, wants "int" or "long"
                    img = img.astype('int')
                hostname = '{}:{}'.format(settings.TF_HOST, settings.TF_PORT)
                req_data = [{'in_tensor_name': settings.TF_TENSOR_NAME,
                             'in_tensor_dtype': floatx,
                             'data': np.expand_dims(img, axis=0)}]
                t = timeit.default_timer()
                client = PredictClient(hostname, model_name, int(model_version))
                self.logger.debug('Created the PredictClient in %s seconds.',
                                  timeit.default_timer() - t)

                prediction = client.predict(req_data, request_timeout=timeout)
                retrying = False
                results = prediction['prediction']
                self.logger.debug('Segmented key %s (model %s:%s, '
                                  'preprocessing: %s, postprocessing: %s)'
                                  ' (%s retries) in %s seconds.',
                                  self._redis_hash, model_name, model_version,
                                  self._redis_values.get('preprocess_function'),
                                  self._redis_values.get('postprocess_function'),
                                  count, timeit.default_timer() - start)
                return results
            except grpc.RpcError as err:
                # pylint: disable=E1101
                retry_statuses = {
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.UNAVAILABLE
                }
                if err.code() in retry_statuses:
                    count += 1
                    # write update to Redis
                    processing_retry_time = time.time() * 1000
                    self.hmset(self._redis_hash, {
                        'number_of_processing_retries': count,
                        'status': 'processing -- RETRY:{} -- {}'.format(
                            count, err.code().name),
                        'timestamp_processing_retry': processing_retry_time,
                        'identity_processing_retry': self.hostname,
                        'timestamp_last_status_update': processing_retry_time
                    })

                    self.logger.warning('%sException `%s: %s` during '
                                        'PredictClient request to model %s:%s.'
                                        'Waiting %s seconds before retrying.',
                                        type(err).__name__, err.code().name,
                                        err.details(), model_name,
                                        model_version, backoff)
                    self.logger.warning('Encountered %s  during PredictClient '
                                        'request to model %s:%s: %s.',
                                        type(err).__name__, model_name,
                                        model_version, err)

                    self.logger.debug('Waiting for %s seconds before retrying',
                                      backoff)
                    time.sleep(backoff)  # sleep before retry
                    retrying = True  # Unneccessary but explicit
                else:
                    retrying = False
                    raise err
            except Exception as err:
                retrying = False
                self.logger.error('Encountered %s during tf-serving request to '
                                  'model %s:%s: %s', type(err).__name__,
                                  model_name, model_version, err)
                raise err

    def _consume(self, redis_hash):
        hvals = self.hgetall(redis_hash)
        # hold on to the redis hash/values for logging purposes
        self._redis_hash = redis_hash
        self._redis_values = hvals
        self.logger.debug('Found hash to process "%s": %s',
                          redis_hash, json.dumps(hvals, indent=4))

        # write update to Redis
        starting_time = time.time() * 1000
        self.hmset(redis_hash, {
            'status': 'started',
            'timestamp_started': starting_time,
            'identity_started': self.hostname,
            'timestamp_last_status_update': starting_time
        })
        model_name = hvals.get('model_name')
        model_version = hvals.get('model_version')
        cuts = hvals.get('cuts', '0')
        field = hvals.get('field_size', '61')

        with utils.get_tempdir() as tempdir:
            fname = self.storage.download(hvals.get('input_file_name'), tempdir)
            image = utils.get_image(fname)

            # configure timeout
            streaming = str(cuts).isdigit() and int(cuts) > 0
            timeout = settings.GRPC_TIMEOUT
            timeout = timeout if not streaming else timeout * int(cuts)

            # Update redis with pre-processing information
            preprocessing_time = time.time() * 1000
            self.hmset(redis_hash, {
                'status': 'pre-processing',
                'timestamp_preprocessing': preprocessing_time,
                'identity_preprocessing': self.hostname,
                'timestamp_last_status_update': preprocessing_time
            })

            pre_funcs = hvals.get('preprocess_function', '').split(',')
            image = self.preprocess(image, pre_funcs, timeout, True)

            # Update redis with prediction information
            predicting_time = time.time() * 1000
            self.hmset(redis_hash, {
                'status': 'predicting',
                'timestamp_predicting': predicting_time,
                'identity_predicting': self.hostname,
                'timestamp_last_status_update': predicting_time
            })

            if streaming:
                image = self.process_big_image(
                    cuts, image, field, model_name, model_version)
            else:
                image = self.grpc_image(
                    image, model_name, model_version, timeout)

            # Update redis with post-processing information
            postprocessing_time = time.time() * 1000
            self.hmset(redis_hash, {
                'status': 'post-processing',
                'timestamp_post-processing': postprocessing_time,
                'identity_post-processing': self.hostname,
                'timestamp_last_status_update': postprocessing_time
            })

            post_funcs = hvals.get('postprocess_function', '').split(',')
            image = self.postprocess(image, post_funcs, timeout, True)

            # write update to Redis
            outputting_time = time.time() * 1000
            self.hmset(redis_hash, {
                'status': 'outputting',
                'timestamp_outputting': outputting_time,
                'identity_outputting': self.hostname,
                'timestamp_last_status_update': outputting_time
            })

            # Save each result channel as an image file
            save_name = hvals.get('original_name', fname)
            subdir = os.path.dirname(save_name.replace(tempdir, ''))
            name = os.path.splitext(os.path.basename(save_name))[0]

            outpaths = utils.save_numpy_array(
                image, name=name, subdir=subdir, output_dir=tempdir)

            # Save each prediction image as zip file
            zip_file = utils.zip_files(outpaths, tempdir)

            # Upload the zip file to cloud storage bucket
            cleaned = zip_file.replace(tempdir, '')
            subdir = os.path.dirname(settings._strip(cleaned))
            subdir = subdir if subdir else None
            dest, output_url = self.storage.upload(zip_file, subdir=subdir)

            # Compute some timings
            output_timestamp = time.time() * 1000
            hash_values = self.hgetall(redis_hash)
            upload_time = float(hash_values.get('timestamp_upload', -1))
            start_time = float(hash_values.get('timestamp_started', -1))
            preprocess_time = float(hash_values.get('timestamp_preprocessing', -1))
            predict_time = float(hash_values.get('timestamp_predicting', -1))
            postprocess_time = float(hash_values.get('timestamp_post-processing', -1))
            outputting_time = float(hash_values.get('timestamp_outputting', -1))
            output_time = float(output_timestamp)
            # compute timing intervals
            upload_to_start_time = (start_time - upload_time) / 1000
            start_to_preprocessing_time = (preprocess_time - start_time) / 1000
            preprocessing_to_predicting_time = (predict_time - preprocess_time) / 1000
            predicting_to_postprocess_time = (postprocess_time - predict_time) / 1000
            postprocess_to_outputting_time = (outputting_time - postprocess_time) / 1000
            outputting_to_output_time = (output_time - outputting_time) / 1000

            # Update redis with the final results
            self.hmset(redis_hash, {
                'identity_output': self.hostname,
                'output_url': output_url,
                'output_file_name': dest,
                'status': self.final_status,
                'timestamp_output': output_timestamp,
                'timestamp_last_status_update': output_timestamp,
                'upload_to_start_time': upload_to_start_time,
                'start_to_preprocessing_time': start_to_preprocessing_time,
                'preprocessing_to_predicting_time': preprocessing_to_predicting_time,
                'predicting_to_postprocess_time': predicting_to_postprocess_time,
                'postprocess_to_outputting_time': postprocess_to_outputting_time,
                'outputting_to_output_time': outputting_to_output_time
            })
            self.logger.debug('Updated status to %s', self.final_status)


class ZipFileConsumer(Consumer):
    """Consumes zip files and uploads the results"""

    def iter_redis_hashes(self, status='new', prefix='predict'):
        """Iterate over hash values in redis.
        Only yield hash values for zip files

        Returns:
            Iterator of all zip hashes with a valid status
        """
        keys = super(ZipFileConsumer, self).iter_redis_hashes(status, prefix)
        for key in keys:
            fname = str(self.hget(key, 'input_file_name'))
            if fname.lower().endswith('.zip'):
                yield key

    def _upload_archived_images(self, hvalues):
        """Extract all image files and upload them to storage and redis"""
        all_hashes = set()
        with utils.get_tempdir() as tempdir:
            fname = self.storage.download(hvalues.get('input_file_name'), tempdir)
            image_files = utils.get_image_files_from_dir(fname, tempdir)
            for imfile in image_files:
                clean_imfile = settings._strip(imfile.replace(tempdir, ''))
                # Save each result channel as an image file
                subdir = os.path.dirname(clean_imfile)
                uploaded_file_path = self.storage.upload(imfile, subdir=subdir)
                new_hash = '{prefix}_{file}_{hash}'.format(
                    prefix=settings.HASH_PREFIX,
                    file=clean_imfile,
                    hash=uuid.uuid4().hex)

                output_timestamp = time.time() * 1000
                new_hvals = dict()
                new_hvals.update(hvalues)
                new_hvals['output_file_name'] = uploaded_file_path
                new_hvals['original_name'] = clean_imfile
                new_hvals['status'] = 'new'
                new_hvals['identity_upload'] = self.hostname
                new_hvals['timestamp_upload'] = output_timestamp
                new_hvals['timestamp_last_status_update'] = output_timestamp
                self.hmset(new_hash, new_hvals)
                self.logger.debug('Added new hash `%s`: %s',
                                  new_hash, json.dumps(new_hvals, indent=4))
                all_hashes.add(new_hash)
        return all_hashes

    def _consume(self, redis_hash):
        start = timeit.default_timer()
        hvals = self.hgetall(redis_hash)
        self.logger.debug('Found hash to process "%s": %s',
                          redis_hash, json.dumps(hvals, indent=4))

        # write update to Redis
        starting_time = time.time() * 1000
        self.hmset(redis_hash, {
            'identity_started': self.hostname,
            'status': 'started',
            'timestamp_started': starting_time,
            'identity_started': self.hostname,
            'timestamp_last_status_update': starting_time
        })
        all_hashes = self._upload_archived_images(hvals)
        self.logger.info('Uploaded %s hashes.  Waiting for ImageConsumers.',
                         len(all_hashes))
        # Now all images have been uploaded with new redis hashes
        # Wait for these to be processed by an ImageFileConsumer
        waiting_time = time.time() * 1000
        self.hmset(redis_hash, {
            'identity_waiting': self.hostname,
            'status': 'waiting',
            'timestamp_waiting': waiting_time,
            'identity_waiting': self.hostname,
            'timestamp_last_status_update': waiting_time
        })

        with utils.get_tempdir() as tempdir:
            finished_hashes = set()
            failed_hashes = set()
            saved_files = set()
            # ping redis until all the sets are finished
            while all_hashes.symmetric_difference(finished_hashes):
                for h in all_hashes:
                    if h in finished_hashes:
                        continue

                    status = self.hget(h, 'status')

                    if status == 'failed':
                        reason = self.hget(h, 'reason')
                        # one of the hashes failed to process
                        self.logger.error('Failed to process hash `%s`: %s',
                                          h, reason)
                        failed_hashes.add(h)
                        finished_hashes.add(h)

                    elif status == self.final_status:
                        # one of our hashes is done!
                        fname = self.hget(h, 'output_file_name')
                        local_fname = self.storage.download(fname, tempdir)
                        self.logger.info('Saved file: %s', local_fname)
                        if zipfile.is_zipfile(local_fname):
                            image_files = utils.get_image_files_from_dir(
                                local_fname, tempdir)
                        else:
                            image_files = [local_fname]

                        for imfile in image_files:
                            saved_files.add(imfile)
                        finished_hashes.add(h)

            if failed_hashes:
                self.logger.warning('Failed to process %s hashes',
                                    len(failed_hashes))

            saved_files = list(saved_files)
            self.logger.info(saved_files)
            zip_file = utils.zip_files(saved_files, tempdir)

            # Upload the zip file to cloud storage bucket
            uploaded_file_path = self.storage.upload(zip_file)

            output_url = self.storage.get_public_url(uploaded_file_path)
            self.logger.debug('Uploaded output to: `%s`', output_url)

            # Update redis with the results
            output_timestamp = time.time() * 1000
            self.hmset(redis_hash, {
                'identity_output': self.hostname,
                'output_url': output_url,
                'output_file_name': uploaded_file_path,
                'status': self.final_status,
                'timestamp_output': output_timestamp,
                'timestamp_last_status_update': output_timestamp
            })
            self.logger.info('Processed all %s images of zipfile `%s` in %s',
                             len(all_hashes), hvals['output_file_name'],
                             timeit.default_timer() - start)
