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

import json
import logging
import os
import tempfile
import timeit
import zipfile
from hashlib import md5
from time import sleep, time

import numpy as np
from PIL import Image
from skimage.external import tifffile as tiff
from keras_preprocessing.image import img_to_array
from tornado import ioloop

from .settings import OUTPUT_DIR


class Consumer(object):
    """Base class for all redis event consumer classes"""

    def __init__(self,
                 redis_client,
                 storage_client,
                 watch_status=None,
                 final_status='done'):
        self.output_dir = OUTPUT_DIR
        self.redis = redis_client
        self.storage = storage_client
        self.watch_status = watch_status
        self.final_status = final_status
        self.logger = logging.getLogger(str(self.__class__.__name__))

    def iter_redis_hashes(self):
        """Iterate over hash values in redis,
        yielding each with a status equal to watch_status
        # Returns: Iterator of all hashes with a valid status
        """
        try:
            keys = self.redis.keys()
            self.logger.debug('Found %s redis keys', len(keys))
        except:
            keys = []

        for key in keys:
            # Check if the key is a hash
            if self.redis.type(key) == 'hash':
                # if watch_status is given, only yield hashes with that status
                if self.watch_status is not None:
                    if self.redis.hget(key, 'status') == self.watch_status:
                        yield key
                else:
                    yield key

    def is_zip_file(self, filename):
        """Returns boolean if cloud file is a zip file
        If using on local file, use ZipFile.is_zipfile instead
        # Arguments:
            filename: key of file in cloud storage
        # Returns:
            True if file is a zip archive otherwise False
        """
        return os.path.splitext(filename)[-1].lower() == '.zip'

    def get_image(self, filepath):
        """Open image file as numpy array
        # Arguments:
            filepath: full filepath of image file
        # Returns:
            img: numpy array of image data
        """
        self.logger.debug('Loading the image into numpy array')
        if os.path.splitext(filepath)[-1].lower() in {'.tif', '.tiff'}:
            img = np.float32(tiff.TiffFile(filepath).asarray())
            # check for channel axis
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
        else:
            img = img_to_array(Image.open(filepath))

        self.logger.debug('Loaded image into numpy array with '
                          'shape %s', img.shape)
        return img

    def save_zip_file(self, files):
        """Save files in zip archive and return the path
        # Arguments:
            files: all filepaths that will be saved in the zip
        # Returns:
            zip_filename: filepath to new zip archive
        """
        try:
            filename = 'prediction_{}'.format(time()).encode('utf-8')
            hashed_filename = '{}.zip'.format(md5(filename).hexdigest())

            zip_filename = os.path.join(self.output_dir, hashed_filename)

            # Create ZipFile and Write each file to it
            with zipfile.ZipFile(zip_filename, 'w') as zip_file:
                for f in files:  # writing each file one by one
                    name = f.replace(self.output_dir, '')
                    if name.startswith(os.path.sep):
                        name = name[1:]

                    zip_file.write(f, arcname=name)
            return zip_filename
        except Exception as err:
            self.logger.error('Failed to write zipfile: %s', err)
            raise err

    def iter_image_files_from_archive(self, zip_path, destination):
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

    def _consume(self):
        raise NotImplementedError

    def consume(self, interval):
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
                self._consume()
            except Exception as err:
                self.logger.error(err)

            sleep(interval)


class PredictionConsumer(Consumer):
    """Consumer to send image data to tf-serving and upload the results"""

    def __init__(self,
                 redis_client,
                 storage_client,
                 tf_client,
                 watch_status='preprocessed',
                 final_status='processed'):
        self.tf_client = tf_client
        super(PredictionConsumer, self).__init__(
            redis_client, storage_client, watch_status, final_status)

    def save_tf_serving_results(self, tf_results, name='', subdir=''):
        """Split complete prediction into components and save each as a tiff
        # Arguments:
            tf_results: numpy array of results from tf-serving
            name: name of original input image file
            subdir: optional subdirectory to save the result.
        # Returns:
            out_paths: list of all saved image paths
        """
        if subdir.startswith(os.path.sep):
            subdir = subdir[1:]

        self.logger.debug('Saving results from tf-serving')
        out_paths = []
        for channel in range(tf_results.shape[-1]):
            try:
                self.logger.debug('saving channel %s', channel)
                if tf_results.ndim >= 4:
                    img = tf_results[:, :, :, channel].astype('float32')
                else:
                    img = tf_results[:, :, channel].astype('float32')

                _name = 'feature_{}.tif'.format(channel)
                if name:
                    _name = '{}_{}'.format(name, _name)

                path = os.path.join(self.output_dir, subdir, _name)

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

    def _iter_cuts(self, img, cuts, field):
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

        for a, b, c, d in self._iter_cuts(img, cuts, field):
            if img.ndim >= 4:
                data = padded_img[:, a:b + 2 * win_x, c:d + 2 * win_y, :]
            else:
                data = padded_img[a:b + 2 * win_x, c:d + 2 * win_y, :]

            images.append(data)
            coords.append((a, b, c, d))

        def post_many():
            timeout = 300 * len(images)
            clients = len(images)
            return self.tf_client.tornado_images(
                images, model_name, model_version,
                timeout=timeout, max_clients=clients)

        predicted = ioloop.IOLoop.current().run_sync(post_many)

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

    def process_image(self, filename, model_name, model_version, cuts=0, field=61):
        """POSTs image data to tf-serving then saves the result
        as a zip and uploads into the cloud bucket.
        # Arguments:
            filename: path to cloud destination of image file
            model_name: name of model in tf-serving
            model_version: integer version number of model in tf-serving
            cuts: if > 1, slices large images and predicts on each slice
            field: receptive field of model
        # Returns:
            output_url: URL of results zip file
        """
        local_fname = self.storage.download(filename)
        img = self.get_image(local_fname)

        if int(cuts) > 1:
            tf_results = self.process_big_image(cuts, img, field, model_name, model_version)
        else:
            # Get tf-serving predictions of image
            tf_results = self.tf_client.post_image(img, model_name, model_version)

        # Save each tf-serving prediction channel as image file
        out_paths = self.save_tf_serving_results(tf_results)

        # Save each prediction image as zip file
        zip_file = self.save_zip_file(out_paths)

        # Upload the zip file to cloud storage bucket
        output_url = self.storage.upload(zip_file)
        self.logger.debug('Saved output to: "%s"', output_url)

        return output_url

    def process_zip(self, filename, model_name, model_version, cuts=0, field=61):
        """Process each image inside a zip archive and save/upload resulting
        zip archive with mirrored folder structure.
        # Arguments:
            filename: path to cloud destination of image file
            model_name: name of model in tf-serving
            model_version: integer version number of model in tf-serving
            cuts: if > 1, slices large images and predicts on each slice
            field: receptive field of model
        # Returns:
            output_url: URL of results zip file
        """
        with tempfile.TemporaryDirectory() as tempdir:
            local_fname = self.storage.download(filename, tempdir)
            if not zipfile.is_zipfile(local_fname):
                self.logger.error('Invalid zip file: %s', local_fname)
                raise ValueError('{} is not a zipfile'.format(local_fname))

            image_files = [f for f in self.iter_image_files_from_archive(local_fname, tempdir)]
            images = (self.get_image(f) for f in image_files)

            def post_many():
                timeout = 300 * len(image_files)
                clients = len(image_files)
                return self.tf_client.tornado_images(
                    images, model_name, model_version,
                    timeout=timeout, max_clients=clients)

            tf_results = ioloop.IOLoop.current().run_sync(post_many)

            all_output = []
            # Save each tf-serving prediction channel as image file
            for results, imfile in zip(tf_results, image_files):
                subdir = os.path.dirname(imfile.replace(tempdir, ''))
                name = os.path.splitext(os.path.basename(imfile))[0]

                _out_paths = self.save_tf_serving_results(results, name=name, subdir=subdir)
                all_output.extend(_out_paths)

        # Save each prediction image as zip file
        zip_file = self.save_zip_file(all_output)

        # Upload the zip file to cloud storage bucket
        upload_dest = self.storage.upload(zip_file)

        return upload_dest

    def _consume(self):
        # process each unprocessed hash
        for redis_hash in self.iter_redis_hashes():
            hash_values = self.redis.hgetall(redis_hash)
            self.logger.debug('Found hash to process "%s": %s',
                              redis_hash, json.dumps(hash_values, indent=4))

            self.redis.hset(redis_hash, 'status', 'processing')
            self.logger.debug('processing image: %s', redis_hash)

            try:
                start = timeit.default_timer()
                fname = hash_values.get('file_name')

                if self.is_zip_file(fname):
                    _func = self.process_zip
                else:
                    _func = self.process_image

                uploaded_file_path = _func(
                    fname,
                    hash_values.get('model_name'),
                    hash_values.get('model_version'),
                    hash_values.get('cuts'))

                output_url = self.storage.get_public_url(uploaded_file_path)
                self.logger.debug('Saved output to: "%s"', output_url)

                self.logger.debug('Processed key %s in %s s',
                                  redis_hash, timeit.default_timer() - start)

                # Update redis with the results
                self.redis.hmset(redis_hash, {
                    'file_path': uploaded_file_path,
                    'output_url': output_url,
                    'status': self.final_status
                })

            except Exception as err:
                # Update redis with failed status
                self.redis.hmset(redis_hash, {
                    'reason': err,
                    'status': 'failed'
                })
                self.logger.error('Failed to process redis key %s. Error: %s',
                                  redis_hash, err)

            except Exception as err:
                self.logger.error('Failed to process redis key %s. Error: %s',
                                  redis_hash, err)

                # Update redis with failed status
                self.redis.hmset(redis_hash, {
                    'reason': err,
                    'status': 'failed'
                })
                else:

                self.logger.debug('Processed key %s in %s s',
                                  redis_hash, timeit.default_timer() - start)

            except Exception as err:
                self.logger.error('Failed to process redis key %s. Error: %s',
                                  redis_hash, err)

