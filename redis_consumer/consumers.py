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

import grpc
import numpy as np

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
        self.logger = logging.getLogger(str(self.__class__.__name__))

    def iter_redis_hashes(self, status='new', prefix='predict'):
        """Iterate over hash values in redis.
        Yield each with the given status value.

        Returns:
            Iterator of all hashes with a valid status
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
        """Consume all redis events every `interval` seconds.

        Args:
            interval: waits this many seconds between consume calls

        Returns:
            nothing: this is the consumer main process
        """
        # process each unprocessed hash
        for redis_hash in self.iter_redis_hashes(status, prefix):
            try:
                start = default_timer()
                self._consume(redis_hash)
                self.logger.debug('Consumed key %s in %ss',
                                  redis_hash, default_timer() - start)
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
            fname = str(self.redis.hget(key, 'file_name'))
            if not fname.lower().endswith('.zip'):
                yield key

    def _process(self, image, key, process_type, timeout=30, streaming=False):
        """Apply each processing function to each image in images.

        Args:
            images: iterable of image data
            key: function to apply to images
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
                results = client.stream_process(req_data, request_timeout=timeout)
            else:
                results = client.process(req_data, request_timeout=timeout)

            self.logger.debug('Finished %s %s-processing image in %ss',
                              key, process_type, default_timer() - start)

            results = results['results']
            # Again, squeeze out batch dimension if unnecessary
            if results.shape[0] == 1:
                results = np.squeeze(results, axis=0)
            return results
        except grpc.RpcError as err:
            retry_statuses = {
                grpc.StatusCode.DEADLINE_EXCEEDED,
                grpc.StatusCode.UNAVAILABLE
            }
            if err.code() in retry_statuses:  # pylint: disable=E1101
                self.logger.warning(err.details())  # pylint: disable=E1101
                self.logger.warning('Encountered %s during %s %s-processing '
                                    'request: %s', type(err).__name__, key,
                                    process_type, err)
                time.sleep((1 + 9 * int(streaming)))  # sleep before retry
                self.logger.debug('Waiting for %s seconds before retrying',
                                  1 + 9 * int(streaming))
                return self._process(image, key, process_type, timeout)
            raise err
        except Exception as err:
            self.logger.error('Encountered %s during %s %s-processing: %s',
                              type(err).__name__, key, process_type, err)
            raise err

    def preprocess(self, image, key, timeout=30, streaming=False):
        """Wrapper for _process_image but can only call with type="pre".

        Args:
            image: numpy array of image data
            key: function to apply to image
            timeout: integer. grpc request timeout.
            streaming: boolean. if True, streams data in multiple requests

        Returns:
            pre-processed image data
        """
        return self._process(image, key, 'pre', timeout, streaming)

    def postprocess(self, image, key, timeout=30, streaming=False):
        """Wrapper for _process_image but can only call with type="post".

        Args:
            image: numpy array of image data
            key: function to apply to image
            timeout: integer. grpc request timeout.
            streaming: boolean. if True, streams data in multiple requests

        Returns:
            post-processed image data
        """
        return self._process(image, key, 'post', timeout, streaming)

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
        start = default_timer()
        self.logger.debug('Segmenting image of shape %s with model %s:%s',
                          img.shape, model_name, model_version)
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
            return prediction['prediction']
        except grpc.RpcError as err:
            retry_statuses = {
                grpc.StatusCode.DEADLINE_EXCEEDED,
                grpc.StatusCode.UNAVAILABLE
            }
            if err.code() in retry_statuses:  # pylint: disable=E1101
                self.logger.warning(err.details())  # pylint: disable=E1101
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
        hvals = self.redis.hgetall(redis_hash)
        self.logger.debug('Found hash to process "%s": %s',
                          redis_hash, json.dumps(hvals, indent=4))

        self.redis.hset(redis_hash, 'status', 'started')

        model_name = hvals.get('model_name')
        model_version = hvals.get('model_version')
        cuts = hvals.get('cuts', '0')
        field = hvals.get('field_size', '61')

        with utils.get_tempdir() as tempdir:
            fname = self.storage.download(hvals.get('file_name'), tempdir)

            start = default_timer()
            image = utils.get_image(fname)

            streaming = str(cuts).isdigit() and int(cuts) > 0
            timeout = settings.GRPC_TIMEOUT
            timeout = timeout if not streaming else timeout * int(cuts)

            self.redis.hset(redis_hash, 'status', 'pre-processing')
            pre = None
            for f in hvals.get('preprocess_function', '').split(','):
                x = pre if pre else image
                pre = self.preprocess(x, f, timeout, streaming)

            self.redis.hset(redis_hash, 'status', 'predicting')
            if streaming:
                prediction = self.process_big_image(
                    cuts, pre, field, model_name, model_version)
            else:
                prediction = self.grpc_image(
                    pre, model_name, model_version, timeout)

            self.redis.hset(redis_hash, 'status', 'post-processing')
            post = None
            for f in hvals.get('postprocess_function', '').split(','):
                x = post if post else prediction
                post = self.postprocess(x, f, timeout, streaming)

            # Save each result channel as an image file
            subdir = os.path.dirname(fname.replace(tempdir, ''))
            name = os.path.splitext(os.path.basename(fname))[0]

            outpaths = utils.save_numpy_array(
                post, name=name, subdir=subdir, output_dir=tempdir)

            self.logger.info('Saved data for image in %ss',
                             default_timer() - start)

            if len(outpaths) > 1:
                # Save each prediction image as zip file
                zip_file = utils.zip_files(outpaths, tempdir)
            else:
                zip_file = outpaths[0]

            # Upload the zip file to cloud storage bucket
            cleaned = zip_file.replace(tempdir, '')
            subdir = os.path.dirname(settings._strip(cleaned))
            subdir = subdir if subdir else None
            uploaded_file_path = self.storage.upload(zip_file, subdir=subdir)

            output_url = self.storage.get_public_url(uploaded_file_path)
            self.logger.debug('Uploaded output to: "%s"', output_url)

            # Update redis with the results
            self.redis.hmset(redis_hash, {
                'output_url': output_url,
                'file_name': uploaded_file_path,
                'status': self.final_status
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
            fname = str(self.redis.hget(key, 'file_name'))
            if fname.lower().endswith('.zip'):
                yield key

    def _upload_archived_images(self, hvalues):
        """Extract all image files and upload them to storage and redis"""
        all_hashes = set()
        with utils.get_tempdir() as tempdir:
            fname = self.storage.download(hvalues.get('file_name'), tempdir)
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

                new_hvals = dict()
                new_hvals.update(hvalues)
                new_hvals['file_name'] = uploaded_file_path
                new_hvals['status'] = 'new'
                self.redis.hmset(new_hash, new_hvals)
                self.logger.debug('Added new hash `%s`: %s',
                                  new_hash, json.dumps(new_hvals, indent=4))
                all_hashes.add(new_hash)
        return all_hashes

    def _consume(self, redis_hash):
        start = default_timer()
        hvals = self.redis.hgetall(redis_hash)
        self.logger.debug('Found hash to process "%s": %s',
                          redis_hash, json.dumps(hvals, indent=4))

        self.redis.hset(redis_hash, 'status', 'started')
        all_hashes = self._upload_archived_images(hvals)
        self.logger.info('Uploaded %s hashes.  Waiting for ImageConsumers.',
                         len(all_hashes))
        # Now all images have been uploaded with new redis hashes
        # Wait for these to be processed by an ImageFileConsumer
        self.redis.hset(redis_hash, 'status', 'waiting')

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
                        fname = self.redis.hget(h, 'file_name')
                        local_fname = self.storage.download(fname, tempdir)
                        self.logger.info('saved file: %s', local_fname)
                        self.logger.info(fname)
                        saved_files.add(local_fname)
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
            self.logger.debug('Uploaded output to: "%s"', output_url)

            # Update redis with the results
            self.redis.hmset(redis_hash, {
                'output_url': output_url,
                'status': self.final_status
            })

            self.logger.info('Processed all %s images of zipfile "%s" in %s',
                             len(all_hashes), hvals['file_name'],
                             default_timer() - start)
