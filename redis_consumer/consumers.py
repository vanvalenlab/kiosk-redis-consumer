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

import datetime
import json
import logging
import os
import sys
import time
import timeit
import urllib
import uuid
import zipfile

import grpc
import numpy as np
import pytz
import skimage

from redis_consumer.grpc_clients import PredictClient
# from redis_consumer.grpc_clients import ProcessClient
from redis_consumer.grpc_clients import TrackingClient
from redis_consumer import utils
from redis_consumer import tracking
from redis_consumer import settings


class Consumer(object):
    """Base class for all redis event consumer classes.

    Args:
        redis_client: obj, Client class to communicate with redis
        storage_client: obj, Client to communicate with cloud storage buckets.
        queue: str, Name of queue to pop off work items.
        final_status: str, Update the status of redis event with this value.
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
        self.final_status = final_status
        self.logger = logging.getLogger(str(self.__class__.__name__))
        self.processing_queue = 'processing-{queue}:{name}'.format(
            queue=self.queue, name=self.hostname)

    def _put_back_hash(self, redis_hash):
        """Put the hash back into the work queue"""
        key = self.redis.rpoplpush(self.processing_queue, self.queue)
        if key is None:
            self.logger.error('RPOPLPUSH got None (%s is empty), key %s was '
                              'removed somehow. Weird!',
                              self.processing_queue, redis_hash)
        elif key != redis_hash:
            self.logger.error('Whoops! RPOPLPUSH sent key %s to %s instead of '
                              '%s. Trying to put it back again.',
                              key, self.queue, redis_hash)
            self._put_back_hash(redis_hash)
        else:
            pass  # success

        # queue_size = self.redis.llen(self.processing_queue)
        # if queue_size == 1:
        #     key = self.redis.rpoplpush(self.processing_queue, self.queue)
        #     if key != redis_hash:
        #         self.logger.warning('`RPOPLPUSH %s %s` popped key %s but'
        #                             'expected key to be %s',
        #                             self.processing_queue, self.queue,
        #                             key, redis_hash)
        #
        # else:
        #     self.logger.warning('Expected `%s` would have 1 item, but has %s. '
        #                         'restarting key `%s` the old way',
        #                         self.processing_queue, queue_size, redis_hash)
        #     res = self.redis.lrem(self.processing_queue, 1, redis_hash)
        #     self.logger.debug('LREM %s got response %s', redis_hash, res)
        #     if res:
        #         self.redis.lpush(self.queue, redis_hash)
        #     else:
        #         self.logger.debug('Trying to put back key %s but it is not in '
        #                           'queue %s', redis_hash, self.processing_queue)

    def get_redis_hash(self):
        while True:

            if self.redis.llen(self.queue) == 0:
                return None

            redis_hash = self.redis.rpoplpush(self.queue, self.processing_queue)

            # if queue is empty, return None
            if redis_hash is None:
                return redis_hash

            self.update_key(redis_hash)  # update timestamp that it was touched

            # if hash is found and valid, return the hash
            if self.is_valid_hash(redis_hash):
                return redis_hash

            # this invalid hash should not be processed by this consumer.
            # remove it from processing, and push it back to the work queue.
            self._put_back_hash(redis_hash)

    def _handle_error(self, err, redis_hash):
        """Update redis with failure information, and log errors.

        Args:
            err: Exception, uncaught error that will be logged.
            redis_hash: string, the hash that will be updated to failure.
        """
        # Update redis with failed status
        self.update_key(redis_hash, {
            'status': 'failed',
            'reason': logging.Formatter().formatException(sys.exc_info()),
        })
        self.logger.error('Failed to process redis key %s due to %s: %s',
                          redis_hash, type(err).__name__, err)

    def is_valid_hash(self, redis_hash):  # pylint: disable=unused-argument
        """Returns True if the consumer should work on the item"""
        return True

    def get_current_timestamp(self):
        """Helper function, returns ISO formatted UTC timestamp"""
        return datetime.datetime.now(pytz.UTC).isoformat()

    def update_key(self, redis_hash, data=None):
        """Update the hash with `data` and updated_by & updated_at stamps.

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
            'updated_at': self.get_current_timestamp(),
            'updated_by': self.hostname,
        })
        self.redis.hmset(redis_hash, data)

    def _consume(self, redis_hash):
        raise NotImplementedError

    def consume(self):
        """Find a redis key and process it"""
        start = timeit.default_timer()
        redis_hash = self.get_redis_hash()

        if redis_hash is not None:  # popped something off the queue
            try:
                self._consume(redis_hash)
            except Exception as err:  # pylint: disable=broad-except
                # log the error and update redis with details
                self._handle_error(err, redis_hash)

            required_fields = [
                'status',
                'model_name',
                'model_version',
                'preprocess_function',
                'postprocess_function',
            ]

            result = self.redis.hmget(redis_hash, *required_fields)
            hvals = {f: v for f, v in zip(required_fields, result)}
            if hvals.get('status') == self.final_status:
                self.logger.debug('Consumed key %s (model %s:%s, '
                                  'preprocessing: %s, postprocessing: %s) '
                                  '(%s retries) in %s seconds.',
                                  redis_hash, hvals.get('model_name'),
                                  hvals.get('model_version'),
                                  hvals.get('preprocess_function'),
                                  hvals.get('postprocess_function'),
                                  0, timeit.default_timer() - start)

            if hvals.get('status') in {self.final_status, 'failed'}:
                # this key is done. remove the key from the processing queue.
                self.redis.lrem(self.processing_queue, 1, redis_hash)

            else:
                # this key is not done yet.
                # remove it from processing and push it back to the work queue.
                self._put_back_hash(redis_hash)

        else:
            self.logger.debug('Queue `%s` is empty. Waiting for %s seconds.',
                              self.queue, settings.EMPTY_QUEUE_TIMEOUT)
            time.sleep(settings.EMPTY_QUEUE_TIMEOUT)


class ImageFileConsumer(Consumer):
    """Consumes image files and uploads the results"""

    def is_valid_hash(self, redis_hash):
        if redis_hash is None:
            return False

        fname = str(self.redis.hget(redis_hash, 'input_file_name'))
        return not fname.lower().endswith('.zip')

    def _get_processing_function(self, process_type, function_name):
        """Based on the function category and name, return the function"""
        clean = lambda x: str(x).lower()
        # first, verify the route parameters
        name = clean(function_name)
        cat = clean(process_type)
        if cat not in settings.PROCESSING_FUNCTIONS:
            raise ValueError('Processing functions are either "pre" or "post" '
                             'processing.  Got %s.' % cat)

        if name not in settings.PROCESSING_FUNCTIONS[cat]:
            if cat not in settings.PROCESSING_FUNCTIONS:
                raise ValueError('"%s" is not a valid %s-processing function'
                                 % (name, cat))
        return settings.PROCESSING_FUNCTIONS[cat][name]

    # def _process(self, image, key, process_type, streaming=False):
    #     """Apply each processing function to image.
    #
    #     Args:
    #         image: numpy array of image data
    #         key: function to apply to image
    #         process_type: pre or post processing
    #         streaming: boolean. if True, streams data in multiple requests
    #
    #     Returns:
    #         list of processed image data
    #     """
    #     # Squeeze out batch dimension if unnecessary
    #     if image.shape[0] == 1:
    #         image = np.squeeze(image, axis=0)
    #
    #     if not key:
    #         return image
    #
    #     self.logger.debug('Starting %s %s-processing image of shape %s',
    #                       key, process_type, image.shape)
    #
    #     retrying = True
    #     count = 0
    #     start = timeit.default_timer()
    #     while retrying:
    #         try:
    #             key = str(key).lower()
    #             process_type = str(process_type).lower()
    #             hostname = '{}:{}'.format(settings.DP_HOST, settings.DP_PORT)
    #             client = ProcessClient(hostname, process_type, key)
    #
    #             if streaming:
    #                 dtype = 'DT_STRING'
    #             else:
    #                 dtype = settings.TF_TENSOR_DTYPE
    #
    #             req_data = [{'in_tensor_name': settings.TF_TENSOR_NAME,
    #                          'in_tensor_dtype': dtype,
    #                          'data': np.expand_dims(image, axis=0)}]
    #
    #             if streaming:
    #                 results = client.stream_process(req_data, settings.GRPC_TIMEOUT)
    #             else:
    #                 results = client.process(req_data, settings.GRPC_TIMEOUT)
    #
    #             finished = timeit.default_timer() - start
    #
    #             self.update_key(self._redis_hash, {
    #                 '{}process_time'.format(process_type): finished
    #             })
    #
    #             self.logger.debug('%s-processed key %s (model %s:%s, '
    #                               'preprocessing: %s, postprocessing: %s)'
    #                               ' (%s retries)  in %s seconds.',
    #                               process_type.capitalize(), self._redis_hash,
    #                               self._redis_values.get('model_name'),
    #                               self._redis_values.get('model_version'),
    #                               self._redis_values.get('preprocess_function'),
    #                               self._redis_values.get('postprocess_function'),
    #                               count, finished)
    #
    #             results = results['results']
    #             # Again, squeeze out batch dimension if unnecessary
    #             if results.shape[0] == 1:
    #                 results = np.squeeze(results, axis=0)
    #
    #             retrying = False
    #             return results
    #         except grpc.RpcError as err:
    #             # pylint: disable=E1101
    #             if err.code() in settings.GRPC_RETRY_STATUSES:
    #                 count += 1
    #                 temp_status = 'retry-processing - {} - {}'.format(
    #                     count, err.code().name)
    #                 self.update_key(self._redis_hash, {
    #                     'status': temp_status,
    #                     'process_retries': count,
    #                 })
    #                 self.logger.warning('%sException `%s: %s` during %s '
    #                                     '%s-processing request.  Waiting %s '
    #                                     'seconds before retrying.',
    #                                     type(err).__name__, err.code().name,
    #                                     err.details(), key, process_type,
    #                                     settings.GRPC_BACKOFF)
    #                 self.logger.debug('Waiting for %s seconds before retrying',
    #                                   settings.GRPC_BACKOFF)
    #                 time.sleep(settings.GRPC_BACKOFF)  # sleep before retry
    #                 retrying = True  # Unneccessary but explicit
    #             else:
    #                 retrying = False
    #                 raise err
    #         except Exception as err:
    #             retrying = False
    #             self.logger.error('Encountered %s during %s %s-processing: %s',
    #                               type(err).__name__, key, process_type, err)
    #             raise err

    def process(self, image, key, process_type):
        start = timeit.default_timer()
        if not key:
            return image

        f = self._get_processing_function(process_type, key)
        results = f(image)

        if results.shape[0] == 1:
            results = np.squeeze(results, axis=0)

        finished = timeit.default_timer() - start

        self.update_key(self._redis_hash, {
            '{}process_time'.format(process_type): finished
        })

        self.logger.debug('%s-processed key %s (model %s:%s, preprocessing: %s,'
                          ' postprocessing: %s) in %s seconds.',
                          process_type.capitalize(), self._redis_hash,
                          self._redis_values.get('model_name'),
                          self._redis_values.get('model_version'),
                          self._redis_values.get('preprocess_function'),
                          self._redis_values.get('postprocess_function'),
                          finished)

        return results

    def preprocess(self, image, keys, streaming=False):
        """Wrapper for _process_image but can only call with type="pre".

        Args:
            image: numpy array of image data
            keys: list of function names to apply to the image
            streaming: boolean. if True, streams data in multiple requests

        Returns:
            pre-processed image data
        """
        pre = None
        for key in keys:
            x = pre if pre else image
            # pre = self._process(x, key, 'pre', streaming)
            pre = self.process(x, key, 'pre')
        return pre

    def postprocess(self, image, keys, streaming=False):
        """Wrapper for _process_image but can only call with type="post".

        Args:
            image: numpy array of image data
            keys: list of function names to apply to the image
            streaming: boolean. if True, streams data in multiple requests

        Returns:
            post-processed image data
        """
        post = None
        for key in keys:
            x = post if post else image
            # post = self._process(x, key, 'post', streaming)
            post = self.process(x, key, 'post')
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

    def grpc_image(self, img, model_name, model_version):
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

                prediction = client.predict(req_data, settings.GRPC_TIMEOUT)
                results = [prediction[k] for k in sorted(prediction.keys())
                           if k.startswith('prediction')]

                if len(results) == 1:
                    results = results[0]

                retrying = False
                finished = timeit.default_timer() - start
                self.update_key(self._redis_hash, {
                    'prediction_time': finished,
                })
                self.logger.debug('Segmented key %s (model %s:%s, '
                                  'preprocessing: %s, postprocessing: %s)'
                                  ' (%s retries) in %s seconds.',
                                  self._redis_hash, model_name, model_version,
                                  self._redis_values.get('preprocess_function'),
                                  self._redis_values.get('postprocess_function'),
                                  count, finished)
                return results
            except grpc.RpcError as err:
                # pylint: disable=E1101
                if err.code() in settings.GRPC_RETRY_STATUSES:
                    count += 1
                    # write update to Redis
                    temp_status = 'retry-predicting - {} - {}'.format(
                        count, err.code().name)
                    self.update_key(self._redis_hash, {
                        'status': temp_status,
                        'predict_retries': count,
                    })
                    self.logger.warning('%sException `%s: %s` during '
                                        'PredictClient request to model %s:%s.'
                                        'Waiting %s seconds before retrying.',
                                        type(err).__name__, err.code().name,
                                        err.details(), model_name,
                                        model_version, settings.GRPC_BACKOFF)
                    self.logger.debug('Waiting for %s seconds before retrying',
                                      settings.GRPC_BACKOFF)
                    time.sleep(settings.GRPC_BACKOFF)  # sleep before retry
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
        start = timeit.default_timer()
        hvals = self.redis.hgetall(redis_hash)
        # hold on to the redis hash/values for logging purposes
        self._redis_hash = redis_hash
        self._redis_values = hvals
        self.logger.debug('Found hash to process `%s` with status `%s`.',
                          redis_hash, hvals.get('status'))

        self.update_key(redis_hash, {
            'status': 'started',
            'identity_started': self.hostname,
        })

        model_name = hvals.get('model_name')
        model_version = hvals.get('model_version')
        cuts = hvals.get('cuts', '0')
        field = hvals.get('field_size', '61')

        with utils.get_tempdir() as tempdir:
            _ = timeit.default_timer()
            fname = self.storage.download(hvals.get('input_file_name'), tempdir)
            image = utils.get_image(fname)

            streaming = str(cuts).isdigit() and int(cuts) > 0

            # Pre-process data before sending to the model
            self.update_key(redis_hash, {
                'status': 'pre-processing',
                'download_time': timeit.default_timer() - _,
            })

            pre_funcs = hvals.get('preprocess_function', '').split(',')
            image = self.preprocess(image, pre_funcs, True)

            # Send data to the model
            self.update_key(redis_hash, {'status': 'predicting'})

            if streaming:
                image = self.process_big_image(
                    cuts, image, field, model_name, model_version)
            else:
                image = self.grpc_image(image, model_name, model_version)

            # Post-process model results
            self.update_key(redis_hash, {'status': 'post-processing'})

            post_funcs = hvals.get('postprocess_function', '').split(',')
            # image[:-1] is targeted at a two semantic head panoptic model
            # TODO This may need to be modified and generalized in the future
            image = self.postprocess(image[:-1], post_funcs, True)

            # Save the post-processed results to a file
            _ = timeit.default_timer()
            self.update_key(redis_hash, {'status': 'saving-results'})

            # Save each result channel as an image file
            save_name = hvals.get('original_name', fname)
            subdir = os.path.dirname(save_name.replace(tempdir, ''))
            name = os.path.splitext(os.path.basename(save_name))[0]

            if isinstance(image, list):
                outpaths = []
                for i in image:
                    outpaths.extend(utils.save_numpy_array(
                        image, name=name, subdir=subdir, output_dir=tempdir))
            else:
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
            self.update_key(redis_hash, {
                'status': self.final_status,
                'output_url': output_url,
                'upload_time': timeit.default_timer() - _,
                'output_file_name': dest,
                'total_jobs': 1,
                'total_time': timeit.default_timer() - start,
                'finished_at': self.get_current_timestamp()
            })


class ZipFileConsumer(Consumer):
    """Consumes zip files and uploads the results"""

    def __init__(self,
                 redis_client,
                 storage_client,
                 queue,
                 final_status='done'):
        # zip files go in a new queue
        zip_queue = '{}-zip'.format(queue)
        self.child_queue = queue
        super(ZipFileConsumer, self).__init__(
            redis_client, storage_client,
            zip_queue, final_status)

    def is_valid_hash(self, redis_hash):
        if redis_hash is None:
            return False

        fname = str(self.redis.hget(redis_hash, 'input_file_name'))
        return fname.lower().endswith('.zip')

    def _upload_archived_images(self, hvalues, redis_hash):
        """Extract all image files and upload them to storage and redis"""
        all_hashes = set()
        with utils.get_tempdir() as tempdir:
            fname = self.storage.download(hvalues.get('input_file_name'), tempdir)
            image_files = utils.get_image_files_from_dir(fname, tempdir)
            for i, imfile in enumerate(image_files):
                clean_imfile = settings._strip(imfile.replace(tempdir, ''))
                # Save each result channel as an image file
                subdir = os.path.dirname(clean_imfile)
                dest, _ = self.storage.upload(imfile, subdir=subdir)

                os.remove(imfile)  # remove the file to save some memory

                new_hash = '{prefix}:{file}:{hash}'.format(
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

                # remove unnecessary/confusing keys (maybe from getting restarted)
                bad_keys = [
                    'children',
                    'children:done',
                    'children:failed',
                    'identity_started',
                ]
                for k in bad_keys:
                    if k in new_hvals:
                        del new_hvals[k]

                self.redis.hmset(new_hash, new_hvals)
                self.redis.lpush(self.child_queue, new_hash)
                self.logger.debug('Added new hash %s: `%s`', i + 1, new_hash)
                self.update_key(redis_hash)
                all_hashes.add(new_hash)
        return all_hashes

    def _upload_finished_children(self, finished_children, redis_hash, expire_time=3600):
        saved_files = set()
        with utils.get_tempdir() as tempdir:
            # process each successfully completed key
            for key in finished_children:
                if not key:
                    continue

                fname = self.redis.hget(key, 'output_file_name')
                if fname is None:
                    if self.redis.exists(key):
                        ttl = self.redis.ttl(key)
                        raise ValueError('Key `%s` exists with TTL %s but has '
                                         'no output_file_name' % (key, ttl))
                    else:
                        raise ValueError('Key `%s` does not exist' % key)
                local_fname = self.storage.download(fname, tempdir)

                self.logger.info('Saved file: %s', local_fname)

                if zipfile.is_zipfile(local_fname):
                    image_files = utils.get_image_files_from_dir(
                        local_fname, tempdir)
                else:
                    image_files = (local_fname,)

                for imfile in image_files:
                    saved_files.add(imfile)

                self.redis.expire(key, expire_time)
                self.update_key(redis_hash)

            # zip up all saved results
            zip_file = utils.zip_files(saved_files, tempdir)

            # Upload the zip file to cloud storage bucket
            path, url = self.storage.upload(zip_file)
            self.logger.debug('Uploaded output to: `%s`', url)
            return path, url

    def _parse_failures(self, failed_children, expire_time=3600):
        failed_hashes = {}
        for key in failed_children:
            if not key:
                continue
            reason = self.redis.hget(key, 'reason')
            # one of the hashes failed to process
            self.logger.error('Failed to process hash `%s`: %s',
                              key, reason)
            failed_hashes[key] = reason
            self.redis.expire(key, expire_time)

        if failed_hashes:
            self.logger.warning('Failed to process hashes: %s',
                                json.dumps(failed_hashes, indent=4))

        # check python2 vs python3
        if hasattr(urllib, 'parse'):
            url_encode = urllib.parse.urlencode  # pylint: disable=E1101
        else:
            url_encode = urllib.urlencode  # pylint: disable=E1101

        return url_encode(failed_hashes)

    def _consume(self, redis_hash):
        start = timeit.default_timer()
        hvals = self.redis.hgetall(redis_hash)
        self.logger.debug('Found hash to process `%s` with status `%s`.',
                          redis_hash, hvals.get('status'))

        key_separator = ','  # char to separate child keys in Redis
        expire_time = 60 * 10  # expire finished child keys in ten minutes

        self.update_key(redis_hash)  # refresh timestamp

        # check to see which child keys have been processed
        children = set(hvals.get('children', '').split(key_separator))
        done = set(hvals.get('children:done', '').split(key_separator))
        failed = set(hvals.get('children:failed', '').split(key_separator))

        if hvals.get('status') == 'new':
            # download the zip file, upload the contents, and enter into Redis
            all_hashes = self._upload_archived_images(hvals, redis_hash)
            self.logger.info('Uploaded %s child keys for key `%s`. Waiting for'
                             ' ImageConsumers.', len(all_hashes), redis_hash)

            # Now all images have been uploaded with new redis hashes
            # Update Redis with child keys and put item back in queue
            self.update_key(redis_hash, {
                'status': 'waiting',
                'children': key_separator.join(all_hashes)
            })

        elif hvals.get('status') == 'waiting':
            # this key was previously processed by a ZipConsumer
            # get keys that have not yet reached a completed status
            remaining_children = children - done - failed
            for child in remaining_children:
                status = self.redis.hget(child, 'status')
                if status == 'failed':
                    failed.add(child)
                elif status == self.final_status:
                    done.add(child)

            remaining_children = children - done - failed

            self.logger.info('Key `%s` has %s children waiting for processing',
                             redis_hash, len(remaining_children))

            # if there are no remaining children, update status to cleanup
            self.update_key(redis_hash, {
                'status': 'cleanup' if not remaining_children else 'waiting',
                'children:done': key_separator.join(d for d in done if d),
                'children:failed': key_separator.join(f for f in failed if f),
            })

        elif hvals.get('status') == 'cleanup':
            # clean up children with status `done` and `failed`

            # get summary data for all finished children
            summary_fields = [
                # 'created_at',
                # 'finished_at',
                'prediction_time',
                'postprocess_time',
                'upload_time',
                'download_time',
                'total_time',
            ]

            summaries = dict()
            for d in done:
                results = self.redis.hmget(d, *summary_fields)
                for field, result in zip(summary_fields, results):
                    try:
                        if field not in summaries:
                            summaries[field] = [float(result)]
                        else:
                            summaries[field].append(float(result))
                    except:
                        self.logger.warning('Summary field `%s` is not a '
                                            'float: %s', field, result)

            for k in summaries:
                summaries[k] = sum(summaries[k]) / len(summaries[k])

            output_file_name, output_url = self._upload_finished_children(
                done, redis_hash, expire_time)

            failures = self._parse_failures(failed, expire_time)

            summaries.update({
                'status': self.final_status,
                'finished_at': self.get_current_timestamp(),
                'output_url': output_url,
                'failures': failures,
                'total_jobs': len(children),
                'output_file_name': output_file_name
            })

            # Update redis with the results
            self.update_key(redis_hash, summaries)

            self.logger.info('Processed all %s images of zipfile `%s` in %s',
                             len(children), hvals.get('input_file_name'),
                             timeit.default_timer() - start)


class TrackingConsumer(Consumer):
    """Consumes some unspecified file format, tracks the images,
       and uploads the results
    """

    def is_valid_hash(self, redis_hash):
        if redis_hash is None:
            return False

        fname = str(self.redis.hget(redis_hash, 'input_file_name')).lower()

        valid_file = (fname.endswith('.trk') or
                      fname.endswith('.trks') or
                      fname.endswith('.tif') or
                      fname.endswith('.tiff'))

        self.logger.debug('Got key %s and decided %s', redis_hash, valid_file)

        return valid_file

    def _get_model(self, redis_hash, hvalues):
        hostname = '{}:{}'.format(settings.TF_HOST, settings.TF_PORT)

        t = timeit.default_timer()
        model = TrackingClient(hostname,
                               redis_hash,
                               hvalues.get('model_name'),
                               int(hvalues.get('model_version')),
                               progress_callback=self._update_progress)

        self.logger.debug('Created the TrackingClient in %s seconds.',
                          timeit.default_timer() - t)
        return model

    def _get_tracker(self, redis_hash, hvalues, raw, segmented):
        tracking_model = self._get_model(redis_hash, hvalues)

        features = {'appearance', 'distance', 'neighborhood', 'regionprop'}
        tracker = tracking.cell_tracker(raw, segmented,
                                        tracking_model,
                                        max_distance=settings.MAX_DISTANCE,
                                        track_length=settings.TRACK_LENGTH,
                                        division=settings.DIVISION,
                                        birth=settings.BIRTH,
                                        death=settings.DEATH,
                                        neighborhood_scale_size=settings.NEIGHBORHOOD_SCALE_SIZE,
                                        features=features)

        self.logger.debug('Created tracker!')
        return tracker

    def _update_progress(self, redis_hash, progress):
        self.update_key(redis_hash, {
            'progress': progress,
        })

    def _load_data(self, hvalues, subdir, fname):
        """
        Given the upload location `input_file_name`, and the downloaded
        location of the same file in subdir/fname, return the raw and annotated
        data. If the input is only raw data, we call up the ImageFileConsumer
        to predict and annotate the data.

        Args:
            hvalues: map of original hvalues of the tracking job
            subdir: string of path that contains the downloaded file
            fname: string of file name inside subdir
        """
        if fname.endswith('.trk') or fname.endswith('.trks'):
            return utils.load_track_file(os.path.join(subdir, fname))

        if not fname.endswith('.tiff') and not fname.endswith('.tif'):
            raise ValueError('_load_data takes in only .tiff, .trk, or .trks')

        # push a key per frame and let ImageFileConsumers segment
        raw = utils.get_image(os.path.join(subdir, fname))

        # remove the last dimensions added by `get_image`

        tiff_stack = np.squeeze(raw, -1)
        if len(tiff_stack.shape) != 3:
            raise ValueError("This tiff file has shape {}, which is not 3 "
                             "dimensions. Tracking can only be done on images "
                             "with 3 dimensions, (time, width, height)".format(
                                 tiff_stack.shape))

        num_frames = len(tiff_stack)
        hash_to_frame = {}
        remaining_hashes = set()

        self.logger.debug('Got tiffstack shape %s.', tiff_stack.shape)
        self.logger.debug('tiffstack num_frames %s.', num_frames)

        with utils.get_tempdir() as tempdir:
            for (i, img) in enumerate(tiff_stack):
                # make a file name for this frame
                segment_fname = '{}-tracking-frame-{}.tif'.format(
                    hvalues.get('original_name'), i)
                segment_local_path = os.path.join(tempdir, segment_fname)

                # upload it
                skimage.external.tifffile.imsave(segment_local_path, img)
                upload_file_name, upload_file_url = self.storage.upload(
                    segment_local_path)

                # prepare hvalues for this frame's hash
                # TODO: model info should not be hardcoded
                current_timestamp = self.get_current_timestamp()
                frame_hvalues = {
                    'identity_upload': self.hostname,
                    'input_file_name': upload_file_name,
                    'original_name': segment_fname,
                    'model_name': settings.MODEL_NAME,
                    'model_version': settings.MODEL_VERSION,
                    'postprocess_function': settings.POSTPROCESS_FUNCTION,
                    'cuts': settings.CUTS,
                    'status': 'new',
                    'created_at': current_timestamp,
                    'updated_at': current_timestamp,
                    'url': upload_file_url
                }

                self.logger.debug("Setting %s", frame_hvalues)

                # make a hash for this frame
                segment_hash = '{prefix}:{file}:{hash}'.format(
                    prefix='predict',
                    file=segment_fname,
                    hash=uuid.uuid4().hex)

                # push the hash to redis and the predict queue
                self.redis.hmset(segment_hash, frame_hvalues)
                self.redis.lpush('predict', segment_hash)
                self.logger.debug('Added new hash for segmentation `%s`: %s',
                                  segment_hash, json.dumps(frame_hvalues,
                                                           indent=4))
                hash_to_frame[segment_hash] = i
                remaining_hashes.add(segment_hash)

            # pop hash, check it, and push it back if it's not done
            # this checks the same hash over and over again, since set's
            # pop is not random. This is fine, since we still need every
            # hash to finish before doing anything.
            frames = {}
            while remaining_hashes:
                finished_hashes = set()

                self.logger.debug('Checking on hashes.')
                for segment_hash in remaining_hashes:
                    status = self.redis.hget(segment_hash, 'status')

                    self.logger.debug('Hash %s has status %s',
                                      segment_hash, status)

                    if status == 'failed':
                        reason = self.redis.hget(segment_hash, 'reason')
                        raise RuntimeError(
                            'Tracking failed during segmentation on frame {}.'
                            '\nSegmentation Error: {}'.format(
                                hash_to_frame[segment_hash], reason))

                    elif status == self.final_status:
                        # if it's done, save the frame, as they'll be packed up
                        # later
                        frame_zip = self.storage.download(
                            self.redis.hget(segment_hash, 'output_file_name'),
                            tempdir)

                        frame_files = list(utils.iter_image_archive(frame_zip,
                                                                    tempdir))

                        if len(frame_files) != 1:
                            raise RuntimeError(
                                "After unzipping predicted frame, got "
                                "back multiple files {}. Expected a "
                                "single file.".format(frame_files))

                        frame_idx = hash_to_frame[segment_hash]
                        frames[frame_idx] = utils.get_image(frame_files[0])
                        finished_hashes.add(segment_hash)

                remaining_hashes -= finished_hashes
                time.sleep(settings.INTERVAL)

        frames = [frames[i] for i in range(num_frames)]

        return {"X": raw, "y": np.array(frames)}

    def _consume(self, redis_hash):
        hvalues = self.redis.hgetall(redis_hash)
        self.logger.debug('Found `%s:*` hash to process "%s": %s',
                          self.queue, redis_hash, json.dumps(hvalues, indent=4))

        # Set status and initial progress
        self.update_key(redis_hash, {
            'status': 'started',
            'progress': 0,
        })

        with utils.get_tempdir() as tempdir:
            fname = self.storage.download(hvalues.get('input_file_name'),
                                          tempdir)
            data = self._load_data(hvalues, tempdir, fname)

            self.logger.debug('Got contents tracking file contents.')
            self.logger.debug('X shape: %s', data['X'].shape)
            self.logger.debug('y shape: %s', data['y'].shape)

            tracker = self._get_tracker(redis_hash, hvalues,
                                        data["X"], data["y"])
            self.logger.debug('Trying to track...')

            tracker._track_cells()

            self.logger.debug('Tracking done!')

            # Save tracking result and upload
            save_name = os.path.join(
                tempdir, hvalues.get('original_name', fname)) + '.trk'

            tracker.dump(save_name, file_format='.trk')
            output_file_name, output_url = self.storage.upload(save_name)

            self.update_key(redis_hash, {
                'status': self.final_status,
                'output_url': output_url,
                'output_file_name': output_file_name,
                'finished_at': self.get_current_timestamp(),
            })
