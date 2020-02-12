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
import logging
import sys
import time
import timeit

import grpc
import numpy as np
import pytz

from redis_consumer.grpc_clients import PredictClient
from redis_consumer import utils
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
                 final_status='done',
                 failed_status='failed'):
        self.output_dir = settings.OUTPUT_DIR
        self.hostname = settings.HOSTNAME
        self.redis = redis_client
        self.storage = storage_client
        self.queue = str(queue).lower()
        self.final_status = final_status
        self.failed_status = failed_status
        self.finished_statuses = {final_status, failed_status}
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

            # hash is invalid. it should not be in this queue.
            self.logger.warning('Found invalid hash in %s: `%s` with '
                                'hvals: %s', self.queue, redis_hash,
                                self.redis.hgetall(redis_hash))
            # self.redis.lrem(self.processing_queue, 1, redis_hash)
            self._put_back_hash(redis_hash)

    def _handle_error(self, err, redis_hash):
        """Update redis with failure information, and log errors.

        Args:
            err: Exception, uncaught error that will be logged.
            redis_hash: string, the hash that will be updated to failure.
        """
        # Update redis with failed status
        self.update_key(redis_hash, {
            'status': self.failed_status,
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

    def purge_processing_queue(self):
        """Move all items from the processing queue to the work queue"""
        while True:
            key = self.redis.rpoplpush(self.processing_queue, self.queue)
            if key is None:
                break
            self.logger.debug('Found stranded key `%s` in queue `%s`. '
                              'Moving it back to `%s`.',
                              key, self.processing_queue, self.queue)

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

        # Purge the processing queue in case of stranded keys
        self.purge_processing_queue()

        redis_hash = self.get_redis_hash()

        if redis_hash is not None:  # popped something off the queue
            try:
                status = self._consume(redis_hash)
            except Exception as err:  # pylint: disable=broad-except
                # log the error and update redis with details
                self._handle_error(err, redis_hash)
                status = self.failed_status

            if status == self.final_status:
                required_fields = [
                    'model_name',
                    'model_version',
                    'preprocess_function',
                    'postprocess_function',
                ]
                result = self.redis.hmget(redis_hash, *required_fields)
                hvals = {f: v for f, v in zip(required_fields, result)}
                self.logger.debug('Consumed key %s (model %s:%s, '
                                  'preprocessing: %s, postprocessing: %s) '
                                  '(%s retries) in %s seconds.',
                                  redis_hash, hvals.get('model_name'),
                                  hvals.get('model_version'),
                                  hvals.get('preprocess_function'),
                                  hvals.get('postprocess_function'),
                                  0, timeit.default_timer() - start)

            if status in self.finished_statuses:
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


class TensorFlowServingConsumer(Consumer):
    """Adds tf-serving basic functionality for predict calls"""

    def __init__(self,
                 redis_client,
                 storage_client,
                 queue,
                 final_status='done'):
        # Create some attributes only used during consume()
        self._redis_hash = None
        self._redis_values = dict()
        super(TensorFlowServingConsumer, self).__init__(
            redis_client, storage_client,
            queue, final_status)

    def _consume(self, redis_hash):
        raise NotImplementedError

    def _get_predict_client(self, model_name, model_version):
        t = timeit.default_timer()
        hostname = '{}:{}'.format(settings.TF_HOST, settings.TF_PORT)
        client = PredictClient(hostname, model_name, int(model_version))
        self.logger.debug('Created the PredictClient in %s seconds.',
                          timeit.default_timer() - t)
        return client

    def grpc_image(self, img, model_name, model_version):
        true_failures, count = 0, 0
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

                req_data = [{'in_tensor_name': settings.TF_TENSOR_NAME,
                             'in_tensor_dtype': floatx,
                             'data': np.expand_dims(img, axis=0)}]

                client = self._get_predict_client(model_name, model_version)

                prediction = client.predict(req_data, settings.GRPC_TIMEOUT)
                results = [prediction[k] for k in sorted(prediction.keys())
                           if k.startswith('prediction')]

                if len(results) == 1:
                    results = results[0]

                retrying = False

                finished = timeit.default_timer() - start
                if self._redis_hash is not None:
                    self.update_key(self._redis_hash, {
                        'prediction_time': finished,
                        'predict_retries': count,
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
                if true_failures > settings.MAX_RETRY > 0:
                    retrying = False
                    raise RuntimeError('Prediction has failed {} times due to '
                                       'error {}'.format(count, err))
                if err.code() in settings.GRPC_RETRY_STATUSES:
                    count += 1
                    is_true_failure = err.code() != grpc.StatusCode.UNAVAILABLE
                    true_failures += int(is_true_failure)
                    # write update to Redis
                    temp_status = 'retry-predicting - {} - {}'.format(
                        count, err.code().name)
                    if self._redis_hash is not None:
                        self.update_key(self._redis_hash, {
                            'status': temp_status,
                            'predict_retries': count,
                        })
                    self.logger.warning('%sException `%s: %s` during '
                                        'PredictClient request to model %s:%s.'
                                        ' Waiting %s seconds before retrying.',
                                        type(err).__name__, err.code().name,
                                        err.details(), model_name,
                                        model_version, settings.GRPC_BACKOFF)
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

    def detect_scale(self, image):
        start = timeit.default_timer()

        if not settings.SCALE_DETECT_ENABLED:
            self.logger.debug('Scale detection disabled. Scale set to 1.')
            return 1

        # Rescale image for compatibility with scale model
        # TODO Generalize to prevent from breaking on new input data types
        if image.shape[-1] == 1:
            image = np.expand_dims(image, axis=0)
        else:
            image = np.expand_dims(image, axis=-1)

        # Reshape data to match size of data that model was trained on
        # TODO Generalize to support rectangular and other shapes
        size = settings.SCALE_RESHAPE_SIZE
        if (image.shape[1] >= size) and (image.shape[2] >= size):
            image, _ = utils.reshape_matrix(image, image, reshape_size=size)

        model_name, model_version = settings.SCALE_DETECT_MODEL.split(':')

        # Loop over each image in the batch dimension for scale prediction
        # TODO Calculate scale_detect_sample based on batch size
        # Could be based on fraction or sampling a minimum set number of frames
        scales = []
        for i in range(0, image.shape[0], settings.SCALE_DETECT_SAMPLE):
            scales.append(self.grpc_image(image[i], model_name, model_version))

        self.logger.debug('Scale detection complete in %s seconds',
                          timeit.default_timer() - start)
        return np.mean(scales)

    def detect_label(self, image):
        start = timeit.default_timer()
        # Rescale for model compatibility
        # TODO Generalize to prevent from breaking on new input data types
        if image.shape[-1] == 1:
            image = np.expand_dims(image, axis=0)
        else:
            image = np.expand_dims(image, axis=-1)

        # TODO Generalize to support rectangular and other shapes
        size = settings.LABEL_RESHAPE_SIZE
        if (image.shape[1] >= size) and (image.shape[2] >= size):
            image, _ = utils.reshape_matrix(image, image, reshape_size=size)

        model_name, model_version = settings.LABEL_DETECT_MODEL.split(':')

        # Loop over each image in batch
        labels = []
        for i in range(0, image.shape[0], settings.LABEL_DETECT_SAMPLE):
            labels.append(self.grpc_image(image[i], model_name, model_version))

        labels = np.array(labels)
        vote = labels.sum(axis=0)
        maj = vote.max()

        self.logger.debug('Label detection complete %s seconds.',
                          timeit.default_timer() - start)
        return np.where(vote == maj)[-1][0]


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
                    prefix=self.child_queue,
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

    def _get_output_file_name(self, key):
        fname = None
        retries = 3
        for _ in range(retries):
            # sometimes this field is missing, gotta get the truth!
            fname = self.redis._redis_master.hget(key, 'output_file_name')
            if fname is None:
                ttl = self.redis.ttl(key)

                if ttl == -2:
                    raise ValueError('Key `%s` does not exist' % key)

                if ttl != -1:
                    self.logger.warning('Key `%s` has a TTL of %s.'
                                        'Why has it been expired already?',
                                        key, ttl)
                else:
                    self.logger.warning('Key `%s` exists with TTL %s but has'
                                        ' no output_file_name', key, ttl)

                self.redis._update_masters_and_slaves()
                time.sleep(settings.GRPC_BACKOFF)
            else:
                break
        else:
            raise ValueError('Key %s had no value for output_file_name'
                             ' %s times in a row.' % (key, retries))
        return fname

    def _upload_finished_children(self, finished_children, redis_hash):
        # saved_files = set()
        with utils.get_tempdir() as tempdir:
            filename = '{}.zip'.format(
                hashlib.md5(str(time.time()).encode('utf-8')).hexdigest())

            zip_path = os.path.join(tempdir, filename)

            zip_kwargs = {
                'compression': zipfile.ZIP_DEFLATED,
                'allowZip64': True,
            }

            with zipfile.ZipFile(zip_path, 'w', **zip_kwargs) as zf:

                # process each successfully completed key
                for key in finished_children:
                    if not key:
                        continue

                    fname = self._get_output_file_name(key)

                    local_fname = self.storage.download(fname, tempdir)

                    self.logger.info('Saved file: %s', local_fname)

                    if zipfile.is_zipfile(local_fname):
                        image_files = utils.get_image_files_from_dir(
                            local_fname, tempdir)
                    else:
                        image_files = (local_fname,)

                    for imfile in image_files:
                        name = imfile.replace(tempdir, '')
                        name = name[1:] if name.startswith(os.path.sep) else name
                        zf.write(imfile, arcname=name)
                        os.remove(imfile)

                    self.update_key(redis_hash)

            # zip up all saved results
            # zip_path = utils.zip_files(saved_files, tempdir)

            # Upload the zip file to cloud storage bucket
            path, url = self.storage.upload(zip_path)
            self.logger.debug('Uploaded output to: `%s`', url)
            return path, url

    def _parse_failures(self, failed_children):
        failed_hashes = {}
        for key in failed_children:
            if not key:
                continue
            reason = self.redis.hget(key, 'reason')
            # one of the hashes failed to process
            self.logger.error('Child key `%s` failed: %s', key, reason)
            failed_hashes[key] = reason

        if failed_hashes:
            self.logger.warning('%s child keys failed to process: %s',
                                len(failed_hashes),
                                json.dumps(failed_hashes, indent=4))

        # check python2 vs python3
        if hasattr(urllib, 'parse'):
            url_encode = urllib.parse.urlencode  # pylint: disable=E1101
        else:
            url_encode = urllib.urlencode  # pylint: disable=E1101

        return url_encode(failed_hashes)

    def _cleanup(self, redis_hash, children, done, failed):
        start = timeit.default_timer()
        # get summary data for all finished children
        summary_fields = [
            # 'created_at',
            # 'finished_at',
            'prediction_time',
            'postprocess_time',
            'upload_time',
            'download_time',
            'total_time',
            'predict_retries',
        ]

        summaries = dict()
        for d in done:
            # TODO: stale data may still be Null, causing missing results.
            results = self.redis.hmget(d, *summary_fields)
            for field, result in zip(summary_fields, results):
                try:
                    if field not in summaries:
                        summaries[field] = [float(result)]
                    else:
                        summaries[field].append(float(result))
                except:  # pylint: disable=bare-except
                    self.logger.warning('Summary field `%s` is not a '
                                        'float: %s', field, result)

        # array as joined string
        for k in summaries:
            summaries[k] = ','.join(str(s) for s in summaries[k])
            # summaries[k] = sum(summaries[k]) / len(summaries[k])

        output_file_name, output_url = self._upload_finished_children(
            done, redis_hash)

        failures = self._parse_failures(failed)

        t = timeit.default_timer() - start

        summaries.update({
            'status': self.final_status,
            'finished_at': self.get_current_timestamp(),
            'output_url': output_url,
            'failures': failures,
            'total_jobs': len(children),
            'cleanup_time': t,
            'output_file_name': output_file_name
        })

        # Update redis with the results
        self.update_key(redis_hash, summaries)

        expire_time = settings.EXPIRE_TIME
        for key in children:
            self.redis.expire(key, expire_time)

        self.logger.debug('All %s child keys will be expiring in %s '
                          'seconds.', len(children), expire_time)
        self.logger.debug('Cleaned up results in %s seconds.', t)

    def _consume(self, redis_hash):
        start = timeit.default_timer()
        hvals = self.redis.hgetall(redis_hash)
        status = hvals.get('status')

        if status in self.finished_statuses:
            self.logger.warning('Found completed hash `%s` with status %s.',
                                redis_hash, hvals.get('status'))
            return status

        self.logger.debug('Found hash to process `%s` with status `%s`.',
                          redis_hash, hvals.get('status'))

        key_separator = ','  # char to separate child keys in Redis

        self.update_key(redis_hash)  # refresh timestamp

        if status == 'new':
            # download the zip file, upload the contents, and enter into Redis
            all_hashes = self._upload_archived_images(hvals, redis_hash)
            self.logger.info('Uploaded %s child keys for key `%s`. Waiting for'
                             ' ImageConsumers.', len(all_hashes), redis_hash)

            # Now all images have been uploaded with new redis hashes
            # Update Redis with child keys and put item back in queue
            next_status = 'waiting'
            self.update_key(redis_hash, {
                'status': next_status,
                'children': key_separator.join(all_hashes),
                'children_upload_time': timeit.default_timer() - start,
                'progress': 0
            })
            return next_status

        if status == 'waiting':
            # this key was previously processed by a ZipConsumer
            # check to see which child keys have already been processed
            children = set(hvals.get('children', '').split(key_separator))
            done = set(hvals.get('children:done', '').split(key_separator))
            failed = set(hvals.get('children:failed', '').split(key_separator))

            # get keys that have not yet reached a completed status
            remaining_children = children - done - failed
            for child in remaining_children:
                child_status = self.redis.hget(child, 'status')
                if child_status == self.failed_status:
                    failed.add(child)
                elif child_status == self.final_status:
                    done.add(child)

            remaining_children = children - done - failed
            progress = (len(done) + len(failed)) / len(children)

            self.logger.info('Key `%s` has %s children waiting for processing',
                             redis_hash, len(remaining_children))

            # if there are no remaining children, update status to cleanup
            self.update_key(redis_hash, {
                'children:done': key_separator.join(d for d in done if d),
                'children:failed': key_separator.join(f for f in failed if f),
                'progress': min(100, max(0, round(progress * 100)))
            })

            if not remaining_children:
                self._cleanup(redis_hash, children, done, failed)
                return self.final_status

            return status

        self.logger.error('Found strange status for key `%s`: %s.',
                          redis_hash, status)
        return status
