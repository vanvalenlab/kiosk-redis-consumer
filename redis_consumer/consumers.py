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
import tempfile
import zipfile

import numpy as np
from tornado import ioloop

from redis_consumer import utils
from redis_consumer.settings import OUTPUT_DIR


class Consumer(object):  # pylint: disable=useless-object-inheritance
    """Base class for all redis event consumer classes"""

    def __init__(self,
                 redis_client,
                 storage_client,
                 final_status='done'):
        self.output_dir = OUTPUT_DIR
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

    def _handle_error(self, err, redis_hash):
        # Update redis with failed status
        self.redis.hmset(redis_hash, {
            'reason': '{}'.format(err),
            'status': 'failed'
        })
        self.logger.error('Failed to process redis key %s. %s: %s',
                          redis_hash, type(err).__name__, err)

    def iter_image_archive(self, zip_path, destination):
        """Extract all files in archie and yield the paths of all images
        # Arguments:
            zip_path: path to zip archive
            destination: path to extract all images
        # Returns:
            Iterator of all image paths in extracted archive
        """
        archive = zipfile.ZipFile(zip_path, 'r')
        is_valid = lambda x: os.path.splitext(x)[1] and '__MACOSX' not in x
        for info in archive.infolist():
            try:
                extracted = archive.extract(info, path=destination)
                if os.path.isfile(extracted):
                    if is_valid(extracted):
                        yield extracted
            except:  # pylint: disable=bare-except
                self.logger.warning('Could not extract %s', info.filename)

    def get_image_files_from_dir(self, fname, destination=None):
        """Based on the file, returns a list of all images in that file.
        # Arguments:
            fname: file (image or zip file)
            destination: folder to save image files from archive, if applicable
        # Returns:
            list of image file paths
        """
        if zipfile.is_zipfile(fname):
            archive = self.iter_image_archive(fname, destination)
            image_files = [f for f in archive]
        else:
            image_files = [fname]
        return image_files

    async def _consume(self, redis_hash):
        raise NotImplementedError

    async def consume(self, status=None, prefix=None):
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
                await self._consume(redis_hash)
                self.logger.debug('Consumed key %s in %ss',
                                  redis_hash, default_timer() - start)
        except Exception as err:  # pylint: disable=broad-except
            self.logger.error(err)


class PredictionConsumer(Consumer):
    """Consumer to send image data to tf-serving and upload the results"""

    def __init__(self,
                 redis_client,
                 storage_client,
                 dp_client,
                 tf_client,
                 final_status='done'):
        self.tf_client = tf_client
        self.dp_client = dp_client
        super(PredictionConsumer, self).__init__(
            redis_client, storage_client, final_status)

    def _iter_cuts(self, img, cuts):
        crop_x = img.shape[img.ndim - 3] // cuts
        crop_y = img.shape[img.ndim - 2] // cuts
        for i in range(cuts):
            for j in range(cuts):
                a, b = i * crop_x, (i + 1) * crop_x
                c, d = j * crop_y, (j + 1) * crop_y
                yield a, b, c, d

    async def process_big_image(self, cuts, img, field, model_name, model_version):
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

        padded_img = utils.pad_image(img, field)

        tf_results = None  # Channel shape is unknown until first request
        for a, b, c, d in self._iter_cuts(img, cuts):
            data = padded_img[..., a:b + 2 * winx, c:d + 2 * winy, :]
            pred = await self.segment_image(data, model_name, model_version)
            if tf_results is None:
                tf_results = np.zeros(list(img.shape)[:-1] + [pred.shape[-1]]).astype('float16')
                self.logger.debug('Initialized output tensor of shape %s',
                                  tf_results.shape)
            tf_results[..., a:b, c:d, :] = pred[..., winx:-winx, winy:-winy, :]
        return tf_results

    def segment_images(self, images, model_name, model_version):
        """Use the TensorFlowServingClient to segment each image
        # Arguments:
            images: iterator of image data to segment
            model_name: name of model in tf-serving
            model_version: integer version number of model in tf-serving
        # Returns:
            results: list of numpy array of transformed data.
        """
        count = len(images)
        self.logger.info('Segmenting %s image%s with model %s:%s',
                         count, 's' if count > 1 else '',
                         model_name, model_version)
        try:
            start = default_timer()
            url = self.tf_client.get_url(model_name, model_version)

            def post_many():
                return self.tf_client.tornado_images(
                    images, url, timeout=300 * count, max_clients=count)

            results = ioloop.IOLoop.current().run_sync(post_many)
            self.logger.debug('Segmented %s image%s with model %s:%s in %s s',
                              count, 's' if count > 1 else '', model_name,
                              model_version, default_timer() - start)
            return results
        except Exception as err:
            self.logger.error('Encountered %s during tf-serving request to '
                              'model %s:%s: %s', type(err).__name__,
                              model_name, model_version, err)
            raise err

    def _process_images(self, images, keys, process_type):
        """Apply each processing function to each image in images
        # Arguments:
            images: iterable of image data
            keys: list of function names to apply to images
            process_type: pre or post processing
        # Returns:
            list of processed image data
        """
        count = len(images)
        process_type = str(process_type).lower()
        if not isinstance(keys, tuple) and not isinstance(keys, list):
            keys = '' if keys is None else keys
            keys = str(keys).split(',')

        for k in keys:
            start = default_timer()
            if not k:
                continue
            self.logger.debug('Starting %s %s-processing %s image%s',
                              k, process_type, count, 's' if count > 1 else '')
            try:
                url = self.dp_client.get_url(process_type, k)

                def post_many():
                    return self.dp_client.tornado_images(
                        images, url, timeout=300 * count, max_clients=count)

                images = ioloop.IOLoop.current().run_sync(post_many)
            except Exception as err:
                self.logger.error('Encountered %s during %s %s-processing: %s',
                                  type(err).__name__, k, process_type, err)
                raise err
            self.logger.debug('Finished %s %s-processing %s image%s in %s s',
                              k, process_type, count, 's' if count > 1 else '',
                              default_timer() - start)
        return images

    def preprocess_images(self, images, keys):
        """Wrapper for _process_images but can only call with type="pre"
        # Arguments:
            images: iterable of image data
            keys: list of function names to apply to images
        # Returns:
            list of pre-processed image data
        """
        return self._process_images(images, keys, 'pre')

    def postprocess_images(self, images, keys):
        """Wrapper for _process_images but can only call with type="post"
        # Arguments:
            images: iterable of image data
            keys: list of function names to apply to images
        # Returns:
            list of post-processed image data
        """
        return self._process_images(images, keys, 'post')

    def _consume(self, redis_hash):
        self.tf_client.verify_endpoint_liveness(code=404, endpoint='')
        self.dp_client.verify_endpoint_liveness(code=200, endpoint='health')

        hvals = self.redis.hgetall(redis_hash)
        self.logger.debug('Found hash to process "%s": %s',
                          redis_hash, json.dumps(hvals, indent=4))

        self.redis.hset(redis_hash, 'status', 'processing')

        model_name = hvals.get('model_name')
        model_version = hvals.get('model_version')
        cuts = hvals.get('cuts', '0')

        try:
            with tempfile.TemporaryDirectory() as tempdir:
                fname = self.storage.download(hvals.get('file_name'), tempdir)
                image_files = self.get_image_files_from_dir(fname, tempdir)
                # TODO: was generator - was causing StreamClosedError
                images = [utils.get_image(f) for f in image_files]

                preprocessed = self.preprocess_images(
                    images, hvals.get('preprocess_function'))

                # predict
                if cuts.isdigit() and int(cuts) > 0:
                    predicted = []
                    for p in preprocessed:
                        prediction = self.process_big_image(
                            cuts, p,
                            hvals.get('field_size', 61),
                            model_name, model_version)
                        predicted.append(prediction)
                else:
                    predicted = self.segment_images(
                        preprocessed, model_name, model_version)

                postprocessed = self.postprocess_images(
                    predicted, hvals.get('postprocess_function'))

                all_output = []
                # Save each result channel as an image file
                for results, imfile in zip(postprocessed, image_files):
                    subdir = os.path.dirname(imfile.replace(tempdir, ''))
                    name = os.path.splitext(os.path.basename(imfile))[0]

                    _out_paths = utils.save_numpy_array(
                        results, name=name, subdir=subdir, output_dir=tempdir)

                    all_output.extend(_out_paths)

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
