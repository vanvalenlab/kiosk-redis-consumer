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

import uuid
import timeit
import datetime
import os
import json
import time
import logging
import zipfile

import pytz
import grpc
import numpy as np

from redis_consumer.grpc_clients import PredictClient
from redis_consumer.grpc_clients import ProcessClient
from redis_consumer import utils
from redis_consumer import settings


class Consumer(object):
    """Base class for all redis event consumer classes.

    Args:
        redis_client: Client class to communicate with redis
        storage_client: Client to communicate with cloud storage buckets.
        final_status: Update the status of redis event with this value.
    """

    def __init__(self,
                 redis_client,
                 storage_client,
                 queue,
                 final_status='done'):
        self.output_dir = settings.OUTPUT_DIR
        self.hostname = settings.HOSTNAME
        self.redis = redis_client
        self.storage = storage_client
        self.queue = str(queue).lower()
        self.processing_queue = 'processing-{}'.format(self.queue)
        self.final_status = final_status
        self.logger = logging.getLogger(str(self.__class__.__name__))

    def get_redis_hash(self):
        while True:
            redis_hash = self.redis.rpoplpush(self.queue, self.processing_queue)

            # if queue is empty, return None
            if redis_hash is None:
                return redis_hash

            # if hash is found and valid, return the hash
            if self.is_valid_hash(redis_hash):
                return redis_hash

            # this invalid hash should not be processed by this consumer.
            # remove it from processing, and push it back to the work queue.
            self.redis.lrem(self.processing_queue, 1, redis_hash)
            self.redis.lpush(self.queue, redis_hash)

    def _handle_error(self, err, redis_hash):
        """Update redis with failure information, and log errors.

        Args:
            err: Exception, uncaught error that will be logged.
            redis_hash: string, the hash that will be updated to failure.
        """
        # Update redis with failed status
        self.update_status(redis_hash, 'failed', {
            'reason': '{}: {}'.format(type(err).__name__, err),
        })
        self.logger.error('Failed to process redis key %s due to %s: %s',
                          redis_hash, type(err).__name__, err)

    def is_valid_hash(self, redis_hash):  # pylint: disable=unused-argument
        """Returns True if the consumer should work on the item"""
        return True

    def get_current_timestamp(self):
        """Helper function, returns ISO formatted UTC timestamp"""
        return datetime.datetime.now(pytz.UTC).isoformat()

    def update_status(self, redis_hash, status, data=None):
        """Update the status of a the given hash.

        Args:
            redis_hash: string, the hash that will be updated
            status: string, the new status value
            data: dict, optional data to include in the hmset call
        """
        if data is not None and not isinstance(data, dict):
            raise ValueError('`data` must be a dictionary, got {}.'.format(
                type(data).__name__))

        data = {} if data is None else data
        data.update({
            'status': status,
            'updated_at': self.get_current_timestamp()
        })
        self.redis.hmset(redis_hash, data)

    def _consume(self, redis_hash):
        raise NotImplementedError

    def consume(self):
        """Consume all redis events every `interval` seconds.

        Args:
            status: string, only consume hashes where `status` == status.
            prefix: string, only consume hashes that start with `prefix`.

        Returns:
            nothing: this is the consumer main process
        """
        start = timeit.default_timer()
        redis_hash = self.get_redis_hash()

        if redis_hash is not None:  # popped something off the queue
            try:
                self._consume(redis_hash)
                hvals = self.redis.hgetall(redis_hash)
                self.logger.debug('Consumed key %s (model %s:%s, '
                                  'preprocessing: %s, postprocessing: %s) '
                                  '(%s retries) in %s seconds.',
                                  redis_hash, hvals.get('model_name'),
                                  hvals.get('model_version'),
                                  hvals.get('preprocess_function'),
                                  hvals.get('postprocess_function'),
                                  0, timeit.default_timer() - start)
            except Exception as err:  # pylint: disable=broad-except
                # log the error and update redis with details
                self._handle_error(err, redis_hash)

            # remove the key from the processing queue
            self.redis.lrem(self.processing_queue, 1, redis_hash)


class ImageFileConsumer(Consumer):
    """Consumes image files and uploads the results"""

    def is_valid_hash(self, redis_hash):
        if redis_hash is None:
            return False
        fname = str(self.redis.hget(redis_hash, 'input_file_name'))
        is_valid = not fname.lower().endswith('.zip')
        return is_valid

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
                    temp_status = 'retry-processing - {} - {}'.format(
                        count, err.code().name)
                    self.update_status(self._redis_hash, temp_status, {
                        'process_retries': count,
                    })
                    sleeptime = np.random.randint(1, 20)
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
                    temp_status = 'retry-predicting - {} - {}'.format(
                        count, err.code().name)
                    self.update_status(self._redis_hash, temp_status, {
                        'predict_retries': count,
                    })
                    self.logger.warning('%sException `%s: %s` during '
                                        'PredictClient request to model %s:%s.'
                                        'Waiting %s seconds before retrying.',
                                        type(err).__name__, err.code().name,
                                        err.details(), model_name,
                                        model_version, backoff)
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
        hvals = self.redis.hgetall(redis_hash)
        # hold on to the redis hash/values for logging purposes
        self._redis_hash = redis_hash
        self._redis_values = hvals
        self.logger.debug('Found hash to process "%s": %s',
                          redis_hash, json.dumps(hvals, indent=4))

        self.update_status(redis_hash, 'started', {
            'identity_started': self.hostname,
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

            # Pre-process data before sending to the model
            self.update_status(redis_hash, 'pre-processing')

            pre_funcs = hvals.get('preprocess_function', '').split(',')
            image = self.preprocess(image, pre_funcs, timeout, True)

            # Send data to the model
            self.update_status(redis_hash, 'predicting')

            if streaming:
                image = self.process_big_image(
                    cuts, image, field, model_name, model_version)
            else:
                image = self.grpc_image(
                    image, model_name, model_version, timeout)

            # Post-process model results
            self.update_status(redis_hash, 'post-processing')

            post_funcs = hvals.get('postprocess_function', '').split(',')
            image = self.postprocess(image, post_funcs, timeout, True)

            # Save the post-processed results to a file
            self.update_status(redis_hash, 'saving-results')

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

            # Update redis with the final results
            self.update_status(redis_hash, self.final_status, {
                'output_url': output_url,
                'output_file_name': dest,
                'finished_at': self.get_current_timestamp(),
            })


class ZipFileConsumer(Consumer):
    """Consumes zip files and uploads the results"""

    def is_valid_hash(self, redis_hash):
        if redis_hash is None:
            return False
        fname = str(self.redis.hget(redis_hash, 'input_file_name'))
        is_valid = fname.lower().endswith('.zip')
        return is_valid

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
                dest, _ = self.storage.upload(imfile, subdir=subdir)

                new_hash = '{prefix}_{file}_{hash}'.format(
                    prefix=settings.HASH_PREFIX,
                    file=clean_imfile,
                    hash=uuid.uuid4().hex)

                current_timestamp = self.get_current_timestamp()
                new_hvals = dict()
                new_hvals.update(hvalues)
                new_hvals['input_file_name'] = dest
                new_hvals['original_name'] = clean_imfile
                new_hvals['status'] = 'new'
                new_hvals['identity_upload'] = self.hostname
                new_hvals['created_at'] = current_timestamp
                new_hvals['updated_at'] = current_timestamp

                self.redis.hmset(new_hash, new_hvals)
                self.redis.lpush(self.queue, new_hash)
                self.logger.debug('Added new hash `%s`: %s',
                                  new_hash, json.dumps(new_hvals, indent=4))
                all_hashes.add(new_hash)
        return all_hashes

    def _consume(self, redis_hash):
        start = timeit.default_timer()
        hvals = self.redis.hgetall(redis_hash)
        self.logger.debug('Found hash to process `%s`: %s',
                          redis_hash, json.dumps(hvals, indent=4))

        self.update_status(redis_hash, 'started')

        all_hashes = self._upload_archived_images(hvals)
        self.logger.info('Uploaded %s hashes.  Waiting for ImageConsumers.',
                         len(all_hashes))

        # Now all images have been uploaded with new redis hashes
        # Wait for these to be processed by an ImageFileConsumer
        self.update_status(redis_hash, 'waiting')

        with utils.get_tempdir() as tempdir:
            finished_hashes = set()
            failed_hashes = set()
            saved_files = set()
            # ping redis until all the sets are finished
            while all_hashes.symmetric_difference(finished_hashes):
                for h in all_hashes:
                    if h in finished_hashes:
                        continue

                    status = self.redis.hget(h, 'status')

                    if status == 'failed':
                        reason = self.redis.hget(h, 'reason')
                        # one of the hashes failed to process
                        self.logger.error('Failed to process hash `%s`: %s',
                                          h, reason)
                        failed_hashes.add(h)
                        finished_hashes.add(h)

                    elif status == self.final_status:
                        # one of our hashes is done!
                        fname = self.redis.hget(h, 'output_file_name')
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
            self.update_status(redis_hash, self.final_status, {
                'identity_output': self.hostname,
                'finished_at': self.get_current_timestamp(),
                'output_url': output_url,
                'output_file_name': uploaded_file_path
            })
            self.logger.info('Processed all %s images of zipfile `%s` in %s',
                             len(all_hashes), hvals['output_file_name'],
                             timeit.default_timer() - start)
