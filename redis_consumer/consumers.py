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
from time import sleep, time
from timeit import default_timer

import json
import logging
import os
import tempfile
import zipfile

import numpy as np
from PIL import Image
from skimage.external import tifffile as tiff
from keras_preprocessing.image import img_to_array
from tornado import ioloop

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

    def get_image(self, filepath):
        """Open image file as numpy array
        # Arguments:
            filepath: full filepath of image file
        # Returns:
            img: numpy array of image data
        """
        self.logger.debug('Loading %s into numpy array', filepath)
        if os.path.splitext(filepath)[-1].lower() in {'.tif', '.tiff'}:
            img = np.float32(tiff.TiffFile(filepath).asarray())
            # check for channel axis
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
        else:
            img = img_to_array(Image.open(filepath))

        self.logger.debug('Loaded %s into numpy array with '
                          'shape %s', filepath, img.shape)
        return img

    def save_zip_file(self, files, dest=None):
        """Save files in zip archive and return the path
        # Arguments:
            files: all filepaths that will be saved in the zip
            dest: saves zip file to this directory, OUTPUT_DIR by default
        # Returns:
            zip_filename: filepath to new zip archive
        """
        try:
            self.logger.debug('Saving %s files to zip archive', len(files))
            output_dir = self.output_dir if dest is None else dest
            filename = 'prediction_{}'.format(time()).encode('utf-8')
            hashed_filename = '{}.zip'.format(md5(filename).hexdigest())
            zip_filename = os.path.join(output_dir, hashed_filename)
            # Create ZipFile and Write each file to it
            with zipfile.ZipFile(zip_filename, 'w') as zip_file:
                for f in files:  # writing each file one by one
                    name = f.replace(output_dir, '')
                    if name.startswith(os.path.sep):
                        name = name[1:]
                    zip_file.write(f, arcname=name)
            self.logger.debug('Saved %s files to %s', len(files), zip_filename)
            return zip_filename
        except Exception as err:
            self.logger.error('Failed to write zipfile: %s', err)
            raise err

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
            except:
                self.logger.warning('Could not extract %s', info.filename)

    def save_numpy_array(self, arr, name='', subdir='', output_dir=None):
        """Split tensor into channels and save each as a tiff
        # Arguments:
            arr: numpy array of image data
            name: name of original input image file
            subdir: optional subdirectory to save the result.
            output_dir: base directory for features
        # Returns:
            out_paths: list of all saved image paths
        """
        output_dir = self.output_dir if output_dir is None else output_dir
        if subdir.startswith(os.path.sep):
            subdir = subdir[1:]

        out_paths = []
        for channel in range(arr.shape[-1]):
            try:
                self.logger.debug('saving channel %s', channel)
                img = arr[..., channel].astype('float32')

                _name = 'feature_{}.tif'.format(channel)
                if name:
                    _name = '{}_{}'.format(name, _name)

                path = os.path.join(output_dir, subdir, _name)

                # Create subdirs if they do not exist
                if not os.path.isdir(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))

                tiff.imsave(path, img)
                self.logger.debug('saved channel %s to %s', channel, path)
                out_paths.append(path)
            except Exception as err:
                out_paths = []
                self.logger.error('Could not save channel %s as image: %s',
                                  channel, err)
        return out_paths

    def _consume(self, redis_hash):
        raise NotImplementedError

    def consume(self, interval, status=None, prefix=None):
        """Consume all redis events every `interval` seconds
        # Arguments:
            interval: waits this many seconds between consume calls
        # Returns:
            nothing: this is the consumer main process
        """
        if not str(interval).isdigit():
            raise ValueError('Expected `interval` to be a number. '
                             'Got {}'.format(type(interval)))

        while True:
            try:
                # process each unprocessed hash
                for redis_hash in self.iter_redis_hashes(status, prefix):
                    start = default_timer()
                    self._consume(redis_hash)
                    self.logger.debug('Consumed key %s in %s s',
                                      redis_hash, default_timer() - start)
            except Exception as err:  # pylint: disable=broad-except
                self.logger.error(err)

            sleep(interval)


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

    def pad_image(self, image, field):
        """Pad each the input image for proper dimensions when stitiching
        # Arguments:
            image: np.array of image data
            field: receptive field size of model
        # Returns:
            image data padded in the x and y axes
        """
        window = (field - 1) // 2
        # Pad images by the field size in the x and y axes
        pad_width = []
        for i in range(len(image.shape)):
            if i == image.ndim - 3:
                pad_width.append((window, window))
            elif i == image.ndim - 2:
                pad_width.append((window, window))
            else:
                pad_width.append((0, 0))

        return np.pad(image, pad_width, mode='reflect')

    def _iter_cuts(self, img, cuts):
        crop_x = img.shape[img.ndim - 3] // cuts
        crop_y = img.shape[img.ndim - 2] // cuts
        for i in range(cuts):
            for j in range(cuts):
                a, b = i * crop_x, (i + 1) * crop_x
                c, d = j * crop_y, (j + 1) * crop_y
                yield a, b, c, d

    def process_big_image(self, cuts, img, field, model_name, model_version):
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
        win_x, win_y = (field - 1) // 2, (field - 1) // 2

        tf_results = None  # Channel shape is unknown until first request
        padded_img = self.pad_image(img, field)

        images, coords = [], []

        for a, b, c, d in self._iter_cuts(img, cuts):
            if img.ndim >= 4:
                data = padded_img[:, a:b + 2 * win_x, c:d + 2 * win_y, :]
            else:
                data = padded_img[a:b + 2 * win_x, c:d + 2 * win_y, :]

            images.append(data)
            coords.append((a, b, c, d))

        predicted = self.segment_images(
            images, len(images), model_name, model_version)

        for (a, b, c, d), pred in zip(coords, predicted):
            if tf_results is None:
                tf_results = np.zeros(list(img.shape)[:-1] + [pred.shape[-1]])
                self.logger.debug('initialized output tensor of shape %s',
                                  tf_results.shape)

            if pred.ndim >= 4:
                tf_results[:, a:b, c:d, :] = pred[:, win_x:-win_x, win_y:-win_y, :]
            else:
                tf_results[a:b, c:d, :] = pred[win_x:-win_x, win_y:-win_y, :]

        return tf_results

    def segment_images(self, images, count, model_name, model_version):
        """Use the TensorFlowServingClient to segment each image
        # Arguments:
            images: iterator of image data to segment
            count: total count of images
            model_name: name of model in tf-serving
            model_version: integer version number of model in tf-serving
        # Returns:
            results: list of numpy array of transformed data.
        """
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

    def process_images(self, images, count, keys, process_type):
        """Apply each processing function to each image in images
        # Arguments:
            images: iterable of image data
            count: total number of images
            keys: list of function names to apply to images
            process_type: pre or post processing
        # Returns:
            list of processed image data
        """
        process_type = str(process_type).lower()
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
                self.logger.debug('%s %s-processed %s image%s in %s s',
                                  process_type.capitalize(), count,
                                  's' if count > 1 else '', k,
                                  default_timer() - start)
            except Exception as err:
                self.logger.error('Encountered %s during %s %s-processing: %s',
                                  type(err).__name__, k, process_type, err)
                raise err
        return images

    def _consume(self, redis_hash):
        hash_values = self.redis.hgetall(redis_hash)
        self.logger.debug('Found hash to process "%s": %s',
                          redis_hash, json.dumps(hash_values, indent=4))

        self.redis.hset(redis_hash, 'status', 'processing')

        prekeys = hash_values.get('preprocess_function', '').split(',')
        postkeys = hash_values.get('postprocess_function', '').split(',')

        model_name = hash_values.get('model_name')
        model_version = hash_values.get('model_version')

        filename = hash_values.get('file_name')

        cuts = hash_values.get('cuts', '0')
        field_size = hash_values.get('field_size', 61)

        try:
            with tempfile.TemporaryDirectory() as tempdir:
                local_fname = self.storage.download(filename, tempdir)

                if zipfile.is_zipfile(local_fname):
                    archive = self.iter_image_archive(local_fname, tempdir)
                    image_files = [f for f in archive]
                else:
                    image_files = [local_fname]

                images = (self.get_image(f) for f in image_files)
                count = len(image_files)

                # preprocess
                preprocessed = self.process_images(
                    images, count, prekeys, 'pre')

                # predict
                if cuts.isdigit() and int(cuts) > 0:
                    predicted = []
                    for p in preprocessed:
                        prediction = self.process_big_image(
                            cuts, p, field_size, model_name, model_version)
                        predicted.append(prediction)
                else:
                    predicted = self.segment_images(
                        preprocessed, count, model_name, model_version)

                # postprocess
                postprocessed = self.process_images(
                    predicted, count, postkeys, 'post')

                all_output = []
                # Save each result channel as an image file
                for results, imfile in zip(postprocessed, image_files):
                    subdir = os.path.dirname(imfile.replace(tempdir, ''))
                    name = os.path.splitext(os.path.basename(imfile))[0]

                    _out_paths = self.save_numpy_array(
                        results, name=name, subdir=subdir, output_dir=tempdir)

                    all_output.extend(_out_paths)

                # Save each prediction image as zip file
                zip_file = self.save_zip_file(all_output, tempdir)

                # Upload the zip file to cloud storage bucket
                uploaded_file_path = self.storage.upload(zip_file)

            output_url = self.storage.get_public_url(uploaded_file_path)
            self.logger.debug('Uploaded output to: "%s"', output_url)

            # Update redis with the results
            self.redis.hmset(redis_hash, {
                'output_url': output_url,
                'status': self.final_status
            })
            self.logger.debug('updated status to %s', self.final_status)

        except Exception as err:  # pylint: disable=broad-except
            self._handle_error(err, redis_hash)

    def consume(self, interval, status='new', prefix='predict'):
        # verify that tf-serving is ready to accept images
        self.tf_client.verify_endpoint_liveness(code=404, endpoint='')
        self.dp_client.verify_endpoint_liveness(code=200, endpoint='health')
        super(PredictionConsumer, self).consume(interval, status, prefix)
