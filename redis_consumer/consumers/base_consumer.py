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
"""Base Consumer classes to provide structure for custom consumer workflows."""
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

import numpy as np
import pytz

from deepcell_toolbox.utils import tile_image, untile_image

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
                 failed_status='failed',
                 name=settings.HOSTNAME,
                 output_dir=settings.OUTPUT_DIR):
        self.redis = redis_client
        self.storage = storage_client
        self.queue = str(queue).lower()
        self.name = name
        self.output_dir = output_dir
        self.final_status = final_status
        self.failed_status = failed_status
        self.finished_statuses = {final_status, failed_status}
        self.logger = logging.getLogger(str(self.__class__.__name__))
        self.processing_queue = 'processing-{queue}:{name}'.format(
            queue=self.queue, name=self.name)

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
        queue_has_items = True
        while queue_has_items:
            key = self.redis.rpoplpush(self.processing_queue, self.queue)
            queue_has_items = key is not None
            if queue_has_items:
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
            'updated_by': self.name,
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
                hvals = dict(zip(required_fields, result))
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
                time.sleep(settings.DO_NOTHING_TIMEOUT)

        else:  # queue is empty
            self.logger.debug('Queue `%s` is empty. Waiting for %s seconds.',
                              self.queue, settings.EMPTY_QUEUE_TIMEOUT)
            time.sleep(settings.EMPTY_QUEUE_TIMEOUT)


class TensorFlowServingConsumer(Consumer):
    """Adds tf-serving basic functionality for predict calls"""

    def __init__(self,
                 redis_client,
                 storage_client,
                 queue,
                 **kwargs):
        # Create some attributes only used during consume()
        self._redis_hash = None
        self._redis_values = dict()
        super(TensorFlowServingConsumer, self).__init__(
            redis_client, storage_client, queue, **kwargs)

    def _consume(self, redis_hash):
        raise NotImplementedError

    def _get_predict_client(self, model_name, model_version):
        t = timeit.default_timer()
        hostname = '{}:{}'.format(settings.TF_HOST, settings.TF_PORT)
        client = PredictClient(hostname, model_name, int(model_version))
        self.logger.debug('Created the PredictClient in %s seconds.',
                          timeit.default_timer() - t)
        return client

    def grpc_image(self, img, model_name, model_version, model_shape,
                   in_tensor_name='image', in_tensor_dtype='DT_FLOAT'):

        in_tensor_dtype = str(in_tensor_dtype).upper()

        start = timeit.default_timer()
        self.logger.debug('Segmenting image of shape %s with model %s:%s',
                          img.shape, model_name, model_version)

        if len(model_shape) == img.ndim + 1:
            img = np.expand_dims(img, axis=0)

        if in_tensor_dtype == 'DT_HALF':
            # TODO: seems like should cast to "half"
            # but the model rejects the type, wants "int" or "long"
            img = img.astype('int')

        req_data = [{'in_tensor_name': in_tensor_name,
                     'in_tensor_dtype': in_tensor_dtype,
                     'data': img}]

        client = self._get_predict_client(model_name, model_version)

        prediction = client.predict(req_data, settings.GRPC_TIMEOUT)
        results = [prediction[k] for k in sorted(prediction.keys())
                   if k.startswith('prediction')]

        if len(results) == 1:
            results = results[0]

        finished = timeit.default_timer() - start
        if self._redis_hash is not None:
            self.update_key(self._redis_hash, {
                'prediction_time': finished,
            })
        self.logger.debug('Segmented key %s (model %s:%s, '
                          'preprocessing: %s, postprocessing: %s)'
                          ' (%s retries) in %s seconds.',
                          self._redis_hash, model_name, model_version,
                          self._redis_values.get('preprocess_function'),
                          self._redis_values.get('postprocess_function'),
                          0, finished)
        return results

    def parse_model_metadata(self, metadata):
        """Parse the metadata response and return list of input metadata.

        Args:
            metadata (dict): model metadata response

        Returns:
            list: List of metadata objects for each defined input.
        """
        # TODO: handle multiple inputs in a general way.
        all_metadata = []
        for k, v in metadata.items():
            shape = ','.join([d['size'] for d in v['tensorShape']['dim']])
            data = {
                'in_tensor_name': k,
                'in_tensor_dtype': v['dtype'],
                'in_tensor_shape': shape,
            }
            all_metadata.append(data)
        return all_metadata

    def get_model_metadata(self, model_name, model_version):
        """Check Redis for saved model metadata or get from TensorFlow Serving.

        The Consumer prefers to get the model metadata from Redis,
        but if the metadata does not exist or is too stale,
        a TensorFlow Serving request will be made.

        Args:
            model_name (str): The model name to get metadata.
            model_version (int): The model version to get metadata.
        """
        model = '{}:{}'.format(model_name, model_version)
        self.logger.debug('Getting model metadata for model %s.', model)

        response = self.redis.hget(model, 'metadata')

        if response:
            self.logger.debug('Got cached metadata for model %s.', model)
            return self.parse_model_metadata(json.loads(response))

        # No response! The key was expired. Get from TFS and update it.
        start = timeit.default_timer()
        client = self._get_predict_client(model_name, model_version)
        model_metadata = client.get_model_metadata()

        try:
            inputs = model_metadata['metadata']['signature_def']['signatureDef']
            inputs = inputs['serving_default']['inputs']

            finished = timeit.default_timer() - start
            self.logger.debug('Got model metadata for %s in %s seconds.',
                              model, finished)

            self.redis.hset(model, 'metadata', json.dumps(inputs))
            self.redis.expire(model, settings.METADATA_EXPIRE_TIME)
            return self.parse_model_metadata(inputs)
        except (KeyError, IndexError) as err:
            self.logger.error('Malformed metadata: %s', model_metadata)
            raise err

    def _predict_big_image(self,
                           image,
                           model_name,
                           model_version,
                           model_shape,
                           model_input_name='image',
                           model_dtype='DT_FLOAT',
                           untile=True,
                           stride_ratio=0.75):
        """Use tile_image to tile image for the model and untile the results.

        Args:
            image (numpy.array): image data as numpy.
            model_name (str): hosted model to send image data.
            model_version (str): model version to query.
            model_shape (tuple): shape of the model's expected input.
            model_dtype (str): dtype of the model's input array.
            model_input_name (str): name of the model's input array.
            untile (bool): untiles results back to image shape if True.
            stride_ratio (float): amount to overlap between tiles, (0, 1].

        Returns:
            numpy.array: untiled results from the model.
        """
        model_ndim = len(model_shape)
        input_shape = (model_shape[model_ndim - 3], model_shape[model_ndim - 2])

        ratio = (model_shape[model_ndim - 3] / settings.TF_MIN_MODEL_SIZE) * \
                (model_shape[model_ndim - 2] / settings.TF_MIN_MODEL_SIZE) * \
                (model_shape[model_ndim - 1])

        batch_size = int(settings.TF_MAX_BATCH_SIZE // ratio)

        tiles, tiles_info = tile_image(
            np.expand_dims(image, axis=0),
            model_input_shape=input_shape,
            stride_ratio=stride_ratio)

        self.logger.debug('Tiling image of shape %s into shape %s.',
                          image.shape, tiles.shape)

        # max_batch_size is 1 by default.
        # dependent on the tf-serving configuration
        results = []
        for t in range(0, tiles.shape[0], batch_size):
            batch = tiles[t:t + batch_size]
            output = self.grpc_image(
                batch, model_name, model_version, model_shape,
                in_tensor_name=model_input_name,
                in_tensor_dtype=model_dtype)

            if not isinstance(output, list):
                output = [output]

            if len(results) == 0:
                results = output
            else:
                for i, o in enumerate(output):
                    results[i] = np.vstack((results[i], o))

        if not untile:
            image = results
        else:
            image = [untile_image(r, tiles_info, model_input_shape=input_shape)
                     for r in results]

        image = image[0] if len(image) == 1 else image
        return image

    def _predict_small_image(self,
                             image,
                             model_name,
                             model_version,
                             model_shape,
                             model_input_name='image',
                             model_dtype='DT_FLOAT'):
        """Pad an image that is too small for the model, and unpad the results.

        Args:
            img (numpy.array): The too-small image to be predicted with
                model_name and model_version.
            model_name (str): hosted model to send image data.
            model_version (str): model version to query.
            model_shape (tuple): shape of the model's expected input.
            model_input_name (str): name of the model's input array.
            model_dtype (str): dtype of the model's input array.

        Returns:
            numpy.array: unpadded results from the model.
        """
        pad_width = []
        model_ndim = len(model_shape)
        for i in range(image.ndim):
            if i in {image.ndim - 3, image.ndim - 2}:
                if i == image.ndim - 3:
                    diff = model_shape[model_ndim - 3] - image.shape[i]
                else:
                    diff = model_shape[model_ndim - 2] - image.shape[i]

                if diff % 2:
                    pad_width.append((diff // 2, diff // 2 + 1))
                else:
                    pad_width.append((diff // 2, diff // 2))
            else:
                pad_width.append((0, 0))

        self.logger.info('Padding image from shape %s to shape %s.',
                         image.shape, tuple([x + y1 + y2 for x, (y1, y2) in
                                             zip(image.shape, pad_width)]))

        padded_img = np.pad(image, pad_width, 'reflect')
        image = self.grpc_image(padded_img, model_name, model_version,
                                model_shape, in_tensor_name=model_input_name,
                                in_tensor_dtype=model_dtype)

        image = [image] if not isinstance(image, list) else image

        # pad batch_size and frames for each output.
        pad_widths = [pad_width] * len(image)
        for i, im in enumerate(image):
            while len(pad_widths[i]) < im.ndim:
                pad_widths[i].insert(0, (0, 0))

        # unpad results
        image = [utils.unpad_image(i, p) for i, p in zip(image, pad_widths)]
        image = image[0] if len(image) == 1 else image

        return image

    def predict(self, image, model_name, model_version, untile=True):
        start = timeit.default_timer()
        model_metadata = self.get_model_metadata(model_name, model_version)

        # TODO: generalize for more than a single input.
        if len(model_metadata) > 1:
            raise ValueError('Model %s:%s has %s required inputs but was only '
                             'given %s inputs.', model_name, model_version,
                             len(model_metadata), len(image))
        model_metadata = model_metadata[0]

        model_input_name = model_metadata['in_tensor_name']
        model_dtype = model_metadata['in_tensor_dtype']

        model_shape = [int(x) for x in model_metadata['in_tensor_shape'].split(',')]
        model_ndim = len(model_shape)

        if model_ndim != image.ndim + 1:
            raise ValueError('Image of shape {} is incompatible with model '
                             '{}:{} with input shape {}'.format(
                                 image.shape, model_name, model_version,
                                 tuple(model_shape)))

        size_x = model_shape[model_ndim - 3]
        size_y = model_shape[model_ndim - 2]

        size_x = image.shape[image.ndim - 3] if size_x <= 0 else size_x
        size_y = image.shape[image.ndim - 2] if size_y <= 0 else size_y

        self.logger.debug('Calling predict on model %s:%s with input shape %s'
                          ' and dtype %s to segment an image of shape %s.',
                          model_name, model_version, tuple(model_shape),
                          model_dtype, image.shape)

        if (image.shape[image.ndim - 3] < size_x or
                image.shape[image.ndim - 2] < size_y):
            # image is too small for the model, pad the image.
            image = self._predict_small_image(image, model_name, model_version,
                                              model_shape, model_input_name,
                                              model_dtype)
        elif (image.shape[image.ndim - 3] > size_x or
              image.shape[image.ndim - 2] > size_y):
            # image is too big for the model, multiple images are tiled.
            image = self._predict_big_image(image, model_name, model_version,
                                            model_shape, model_input_name,
                                            model_dtype, untile=untile)
        else:
            # image size is perfect, just send it to the model
            image = self.grpc_image(image, model_name, model_version,
                                    model_shape, model_input_name, model_dtype)

        if isinstance(image, list):
            output_shapes = [i.shape for i in image]
        else:
            output_shapes = [image.shape]  # cast as list

        self.logger.debug('Got response from model %s:%s of shape %s in %s '
                          'seconds.', model_name, model_version, output_shapes,
                          timeit.default_timer() - start)

        return image


class ZipFileConsumer(Consumer):
    """Consumes zip files and uploads the results"""

    def __init__(self,
                 redis_client,
                 storage_client,
                 queue,
                 **kwargs):
        # zip files go in a new queue
        zip_queue = '{}-zip'.format(queue)
        self.child_queue = queue
        super(ZipFileConsumer, self).__init__(
            redis_client, storage_client, zip_queue, **kwargs)

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
                new_hvals['identity_upload'] = self.name
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
            filename = '{}.zip'.format(uuid.uuid4().hex)

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
