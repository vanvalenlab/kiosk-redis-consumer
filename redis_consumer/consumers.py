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

from hashlib import md5
from timeit import default_timer

import os
import json
import time
import logging
import zipfile

import grpc
import numpy as np
from redis.exceptions import ConnectionError

from redis_consumer.storage import StorageException
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

    def __init__(self, redis_client, storage_client, final_status='done'):
        self.output_dir = settings.OUTPUT_DIR
        self.redis = redis_client
        self.storage = storage_client
        self.final_status = final_status
        self._redis_retry_timeout = settings.REDIS_TIMEOUT
        self.logger = logging.getLogger(str(self.__class__.__name__))
        self.HOSTNAME = settings.HOSTNAME

    def iter_redis_hashes(self, status='new', prefix='predict'):
        """Iterate over hash values in redis.
        Yield each with the given status value.

        Returns:
            Iterator of all hashes with a valid status
        """
        for key in self._keys():
            # Check if the key is a hash
            if self._redis_type(key) == 'hash':
                # Check if necessary to filter based on prefix
                if prefix is not None:
                    if not key.startswith(str(prefix).lower()):
                        # Wrong prefix, skip it.
                        continue
                # if status is given, only yield hashes with that status
                if status is not None:
                    if self.hget(key, 'status') == str(status):
                        yield key
                else:  # no need to check the status
                    yield key

    def _handle_error(self, err, redis_hash):
        # Update redis with failed status
        failing_time = time.time() * 1000
        failing_dict = {
            'reason': '{}: {}'.format(type(err).__name__, err),
            'status': 'failed',
            'timestamp_failed': failing_time,
            'identity_failed': self.HOSTNAME,
            'timestamp_last_status_update': failing_time
        }
        self.hmset(redis_hash, failing_dict)
        # log update
        self.logger.error('Failed to process redis key %s. %s: %s',
                          redis_hash, type(err).__name__, err)

    def _consume(self, redis_hash):
        raise NotImplementedError

    def _redis_type(self, redis_key):
        while True:
            try:
                response = self.redis.type(redis_key)
                break
            except ConnectionError as err:
                self.logger.warn('Encountered %s: %s when calling redis.type(). '
                                 'Retrying in %s seconds.',
                                 type(err).__name__, err,
                                 self._redis_retry_timeout)
                time.sleep(self._redis_retry_timeout)
        return response

    def _keys(self):
        while True:
            try:
                response = self.redis.keys()
                break
            except ConnectionError as err:
                self.logger.warn('Encountered %s: %s when calling redis.keys(). '
                                 'Retrying in %s seconds.',
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
                self.logger.warn('Encountered %s: %s when calling redis.hset(). '
                                 'Retrying in %s seconds.',
                                 type(err).__name__, err,
                                 self._redis_retry_timeout)
                time.sleep(self._redis_retry_timeout)
        return response

    def hget(self, rhash, key):
        while True:
            try:
                response = self.redis.hget(rhash, key)
                break
            except ConnectionError as err:
                self.logger.warn('Encountered %s: %s when calling redis.hget(). '
                                 'Retrying in %s seconds.',
                                 type(err).__name__, err,
                                 self._redis_retry_timeout)
                time.sleep(self._redis_retry_timeout)
        return response

    def hmset(self, rhash, data):
        while True:
            try:
                response = self.redis.hmset(rhash, data)
                break
            except ConnectionError as err:
                self.logger.warn('Encountered %s: %s when calling redis.hmset(). '
                                 'Retrying in %s seconds.',
                                 type(err).__name__, err,
                                 self._redis_retry_timeout)
                time.sleep(self._redis_retry_timeout)
        return response

    def hgetall(self, rhash):
        while True:
            try:
                response = self.redis.hgetall(rhash)
                break
            except ConnectionError as err:
                self.logger.warn('Encountered %s: %s when calling redis.hgetall(). '
                                 'Retrying in %s seconds.',
                                 type(err).__name__, err,
                                 self._redis_retry_timeout)
                time.sleep(self._redis_retry_timeout)
        return response

    def consume(self, status=None, prefix=None, retries=3):
        """Consume all redis events every `interval` seconds.

        Args:
            interval: waits this many seconds between consume calls

        Returns:
            nothing: this is the consumer main process
        """
        # process each unprocessed hash
        for redis_hash in self.iter_redis_hashes(status, prefix):
            retry_count = 0
            finished = False
            while retry_count < retries and not finished:
                try:
                    start = default_timer()
                    self._consume(redis_hash)
                    self.logger.debug('Consumed key %s in %ss',
                                      redis_hash, default_timer() - start)
                    finished = True
                except StorageException:
                    retry_count = retry_count + 1
                except Exception as err:  # pylint: disable=broad-except
                    self._handle_error(err, redis_hash)
                    finished = True


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

        start = default_timer()
        self.logger.debug('Starting %s %s-processing image of shape %s',
                          key, process_type, image.shape)

        # using while loop instead of recursive call for
        # help with memory footprint issue.
        retrying = True
        count = 0
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

                self.logger.debug('Finished %s %s-processing image in %ss',
                                  key, process_type, default_timer() - start)

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
                if err.code() in retry_statuses:  # pylint: disable=E1101
                    count += 1
                    # write update to Redis
                    processing_retry_time = time.time() * 1000
                    processing_retry_dict = {
                        'number_of_processing_retries': count,
                        'status': 'processing -- RETRY:{} -- {}'.format(
                            count, err.code().name),
                        'timestamp_processing_retry': processing_retry_time,
                        'identity_processing_retry': self.HOSTNAME,
                        'timestamp_last_status_update': processing_retry_time
                    }
                    self.hmset(self._redis_hash, processing_retry_dict)
                    # log processing retry error
                    self.logger.warning(err.details())  # pylint: disable=E1101
                    self.logger.warning('%s during %s %s-processing request: '
                                        '%s', type(err).__name__, key,
                                        process_type, err)
                    sleeptime = np.random.randint(24, 44)
                    sleeptime = 1 + sleeptime * int(streaming)
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
        start = default_timer()
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
                          tf_results.shape, default_timer() - start)
        return tf_results

    def grpc_image(self, img, model_name, model_version, timeout=30):
        count = 0
        start = default_timer()
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
                client = PredictClient(hostname, model_name, int(model_version))
                prediction = client.predict(req_data, request_timeout=timeout)
                self.logger.debug('Segmented image with model %s:%s in %ss',
                                  model_name, model_version,
                                  default_timer() - start)
                retrying = False
                return prediction['prediction']
            except grpc.RpcError as err:
                retry_statuses = {
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.UNAVAILABLE
                }
                if err.code() in retry_statuses:  # pylint: disable=E1101
                    count += 1
                    # write update to Redis
                    processing_retry_time = time.time() * 1000
                    processing_retry_dict = {
                        'number_of_processing_retries': count,
                        'status': 'processing -- RETRY:{} -- {}'.format(
                            count, err.code().name),
                        'timestamp_processing_retry': processing_retry_time,
                        'identity_processing_retry': self.HOSTNAME,
                        'timestamp_last_status_update': processing_retry_time
                    }
                    self.hmset(self._redis_hash, processing_retry_dict)
                    # log processing retry error
                    self.logger.warning(err.details())  # pylint: disable=E1101
                    self.logger.warning('Encountered %s during tf-serving request '
                                        'to model %s:%s: %s', type(err).__name__,
                                        model_name, model_version, err)
                    sleeptime = np.random.randint(9, 20) + 1
                    self.logger.debug('Waiting for %s seconds before retrying',
                                      sleeptime)
                    time.sleep(sleeptime)  # sleep before retry
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
        self._redis_hash = redis_hash
        hvals = self.hgetall(redis_hash)
        self.logger.debug('Found hash to process "%s": %s',
                          redis_hash, json.dumps(hvals, indent=4))

        # write update to Redis
        starting_time = time.time() * 1000
        starting_dict = {
            'status': 'started',
            'timestamp_started': starting_time,
            'identity_started': self.HOSTNAME,
            'timestamp_last_status_update': starting_time
        }
        self.hmset(redis_hash, starting_dict)
        # log update
        self.logger.debug('Updated hash with %s', starting_dict)
        model_name = hvals.get('model_name')
        model_version = hvals.get('model_version')
        cuts = hvals.get('cuts', '0')
        field = hvals.get('field_size', '61')

        with utils.get_tempdir() as tempdir:
            fname = self.storage.download(hvals.get('input_file_name'), tempdir)

            start = default_timer()
            image = utils.get_image(fname)

            # configure timeout
            streaming = str(cuts).isdigit() and int(cuts) > 0
            timeout = settings.GRPC_TIMEOUT
            timeout = timeout if not streaming else timeout * int(cuts)

            # write update to Redis
            preprocessing_time = time.time() * 1000
            preprocessing_dict = {
                'status': 'pre-processing',
                'timestamp_preprocessing': preprocessing_time,
                'identity_preprocessing': self.HOSTNAME,
                'timestamp_last_status_update': preprocessing_time
            }
            self.hmset(redis_hash, preprocessing_dict)

            pre_funcs = hvals.get('preprocess_function', '').split(',')
            image = self.preprocess(image, pre_funcs, timeout, streaming)

            # write update to Redis
            predicting_time = time.time() * 1000
            predicting_dict = {
                'status': 'predicting',
                'timestamp_predicting': predicting_time,
                'identity_predicting': self.HOSTNAME,
                'timestamp_last_status_update': predicting_time
            }
            self.hmset(redis_hash, predicting_dict)

            if streaming:
                image = self.process_big_image(
                    cuts, image, field, model_name, model_version)
            else:
                image = self.grpc_image(
                    image, model_name, model_version, timeout)

            # write update to Redis
            postprocessing_time = time.time() * 1000
            postprocessing_dict = {
                'status': 'post-processing',
                'timestamp_post-processing': postprocessing_time,
                'identity_post-processing': self.HOSTNAME,
                'timestamp_last_status_update': postprocessing_time
            }
            self.hmset(redis_hash, postprocessing_dict)

            post_funcs = hvals.get('postprocess_function', '').split(',')
            image = self.postprocess(image, post_funcs, timeout, streaming)

            # write update to Redis
            outputting_time = time.time() * 1000
            outputting_dict = {
                'status': 'outputting',
                'timestamp_outputting': outputting_time,
                'identity_outputting': self.HOSTNAME,
                'timestamp_last_status_update': outputting_time
            }
            self.hmset(redis_hash, outputting_dict)

            # Save each result channel as an image file
            save_name = hvals.get('original_name', fname)
            subdir = os.path.dirname(save_name.replace(tempdir, ''))
            name = os.path.splitext(os.path.basename(save_name))[0]

            outpaths = utils.save_numpy_array(
                image, name=name, subdir=subdir, output_dir=tempdir)

            self.logger.info('Saved data for image in %ss',
                             default_timer() - start)

            # Save each prediction image as zip file
            zip_file = utils.zip_files(outpaths, tempdir)

            # Upload the zip file to cloud storage bucket
            cleaned = zip_file.replace(tempdir, '')
            subdir = os.path.dirname(settings._strip(cleaned))
            subdir = subdir if subdir else None
            dest, output_url = self.storage.upload(zip_file, subdir=subdir)
            self.logger.debug('Uploaded output to: "%s"', output_url)

            # Compute some timings
            output_timestamp = time.time() * 1000
            hash_values = self.redis_hgetall(redis_hash)
            upload_time = float(hash_values['timestamp_upload'])
            start_time = float(hash_values['timestamp_started'])
            preprocess_time = float(hash_values['timestamp_preprocessing'])
            predict_time = float(hash_values['timestamp_predicting'])
            postprocess_time = float(hash_values['timestamp_post-processing'])
            outputting_time = float(hash_values['timestamp_outputting'])
            output_time = float(output_timestamp)
            # compute timing intervals
            upload_to_start_time = (start_time - upload_time) / 1000
            start_to_preprocessing_time = (preprocess_time - start_time) / 1000
            preprocessing_to_predicting_time = \
                    (predict_time - preprocess_time) / 1000
            predicting_to_postprocess_time = \
                    (postprocess_time - predict_time) / 1000
            postprocess_to_outputting_time = \
                    (outputting_time - postprocess_time) / 1000
            outputting_to_output_time = (output_time - outputting_time) / 1000
            # Update redis with the final results
            output_dict = {
                'identity_output': self.HOSTNAME,
                'output_url': output_url,
                'output_file_name': dest,
                'status': self.final_status,
                'timestamp_output': output_timestamp,
                'timestamp_last_status_update': output_timestamp,
                'upload_to_start_time': upload_to_start_time,
                'start_to_preprocessing_time': start_to_preprocessing_time,
                'preprocessing_to_predicting_time': \
                        preprocessing_to_predicting_time,
                'predicting_to_postprocess_time': \
                        predicting_to_postprocess_time,
                'postprocess_to_outputting_time': \
                        postprocess_to_outputting_time,
                'outputting_to_output_time': outputting_to_output_time
            }
            self.hmset(redis_hash, output_dict)
            # log status update
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
                    hash=md5(str(time.time()).encode('utf-8')).hexdigest())

                output_timestamp = time.time() * 1000
                new_hvals = dict()
                new_hvals.update(hvalues)
                new_hvals['output_file_name'] = uploaded_file_path
                new_hvals['original_name'] = clean_imfile
                new_hvals['status'] = 'new'
                new_hvals['identity_upload'] = self.HOSTNAME
                new_hvals['timestamp_upload'] = output_timestamp
                new_hvals['timestamp_last_status_update'] = output_timestamp
                self.hmset(new_hash, new_hvals)
                self.logger.debug('Added new hash `%s`: %s',
                                  new_hash, json.dumps(new_hvals, indent=4))
                all_hashes.add(new_hash)
        return all_hashes

    def _consume(self, redis_hash):
        start = default_timer()
        hvals = self.hgetall(redis_hash)
        self.logger.debug('Found hash to process "%s": %s',
                          redis_hash, json.dumps(hvals, indent=4))

        # write update to Redis
        starting_time = time.time() * 1000
        starting_dict = {
            'identity_started': self.HOSTNAME,
            'status': 'started',
            'timestamp_started': starting_time,
            'identity_started': self.HOSTNAME,
            'timestamp_last_status_update': starting_time
        }
        self.hmset(redis_hash, starting_dict)
        # log update
        self.logger.debug('Updated hash with %s', starting_dict)
        all_hashes = self._upload_archived_images(hvals)
        self.logger.info('Uploaded %s hashes.  Waiting for ImageConsumers.',
                         len(all_hashes))
        # Now all images have been uploaded with new redis hashes
        # Wait for these to be processed by an ImageFileConsumer
        waiting_time = time.time() * 1000
        waiting_dict = {
            'identity_waiting': self.HOSTNAME,
            'status': 'waiting',
            'timestamp_waiting': waiting_time,
            'identity_waiting': self.HOSTNAME,
            'timestamp_last_status_update': waiting_time
        }
        self.hmset(redis_hash, waiting_dict)

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
            output_dict = {
                'identity_output': self.HOSTNAME,
                'output_url': output_url,
                'output_file_name': uploaded_file_path,
                'status': self.final_status,
                'timestamp_output': output_timestamp,
                'timestamp_last_status_update': output_timestamp
            }
            self.hmset(redis_hash, output_dict)
            # log status update
            self.logger.info('Processed all %s images of zipfile `%s` in %s',
                             len(all_hashes), hvals['output_file_name'],
                             default_timer() - start)
