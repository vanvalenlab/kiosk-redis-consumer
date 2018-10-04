# Copyright 2016-2018 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/kiosk-consumer/LICENSE
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
import zipfile
from hashlib import md5
from time import sleep, time

import numpy as np
from PIL import Image
from skimage.external import tifffile as tiff
from tensorflow.python.keras.preprocessing.image import img_to_array

from .settings import DOWNLOAD_DIR, OUTPUT_DIR


class Consumer(object):

    def __init__(self, redis_client, storage_client):
        self.redis = redis_client
        self.storage = storage_client
        self.logger = logging.getLogger(str(self.__class__.__name__))
    
    def iter_redis_hashes(self):
        """Iterate over all unprocessed keys in redis"""
        try:
            keys = self.redis.keys()
            self.logger.debug('Found %s redis keys', len(keys))
        except:
            keys = []
        
        for key in keys:
            # Check if the key is a hash
            if self.redis.type(key) == 'hash':
                yield key

    def consume(self):
        raise NotImplementedError


class PredictionConsumer(Consumer):

    def __init__(self, redis_client, storage_client, tf_client):
        self.tf_client = tf_client
        self.output_dir = OUTPUT_DIR
        super(PredictionConsumer, self).__init__(redis_client, storage_client)
    
    def iter_redis_hashes(self):
        """Iterate over all hashes, yield unprocessed tf-serving events"""
        for h in super(PredictionConsumer, self).iter_redis_hashes():
            # Check if the hash has been claimed by a tf-serving instance
            if self.redis.hget(h, 'processed') == 'no':
                yield h

    def save_tf_serving_results(self, tf_results):
        """Split complete prediction into components and save each as a tiff
        TODO: this looks to only work for 2D data
        """
        self.logger.debug('Saving results from tf-serving')
        out_paths = []
        for channel in range(tf_results.shape[-1]):
            try:
                self.logger.debug('saving channel %s', channel)
                if tf_results.ndim >= 4:
                    img = tf_results[:, :, :, channel].astype('float32')
                else:
                    img = tf_results[:, :, channel].astype('float32')
                path = os.path.join(self.output_dir, 'feature_{}.tif'.format(channel))
                tiff.imsave(path, img)
                self.logger.debug('saved channel %s to %s', channel, path)
                out_paths.append(path)
            except Exception as err:
                out_paths = []
                self.logger.error('Could not save channel %s as image: %s',
                    channel, err)
        return out_paths

    def save_zip_file(self, out_paths):
        """Save output images as tiff files and return their paths"""
        try:
            filename = 'prediction_{}'.format(time()).encode('utf-8')
            hashed_filename = '{}.zip'.format(md5(filename).hexdigest())

            zip_filename = os.path.join(self.output_dir, hashed_filename)

            # Create ZipFile and Write tiff files to it
            with zipfile.ZipFile(zip_filename, 'w') as zip_file:
                # writing each file one by one
                for out_file in out_paths:
                    zip_file.write(out_file, arcname=os.path.basename(out_file))
            return zip_filename
        except Exception as err:
            self.logger.error('Failed to write zipfile: %s', err)
            raise err

    def process_image(self, filename, storage_url, model_name, model_version, cuts=0, field=61):
        """POSTS image data to tf-serving then saves the result
        as a zip and uploads into the cloud bucket.
        # Arguments:
            filename: path to cloud destination of image file
            storage_url: URL of file in cloud storage bucket
            model_name: name of model in tf-serving
            model_version: integer version number of model in tf-serving
        # Returns:
            output_url: URL of results zip file
        """
        downloaded_filepath = self.storage.download(filename, storage_url)
        self.logger.debug('Loading the image into numpy array')
        if os.path.splitext(downloaded_filepath)[-1].lower() in {'.tif', '.tiff'}:
            img = np.float32(tiff.TiffFile(downloaded_filepath).asarray())
        else:
            img = img_to_array(Image.open(downloaded_filepath))
        self.logger.debug('Loaded image into numpy array with '
                          'shape %s', img.shape)
        cuts = int(cuts)
        if cuts > 1:
            crop_x = img.shape[img.ndim - 3] // cuts
            crop_y = img.shape[img.ndim - 2] // cuts
            win_x, win_y = (field - 1) // 2, (field - 1) // 2

            tf_results = None
            pad_width = []
            for i in range(len(img.shape)):
                if i == img.ndim - 3:
                    pad_width.append((win_x, win_x))
                elif i == img.ndim - 2:
                    pad_width.append((win_y, win_y))
                else:
                    pad_width.append((0, 0))

            padded_img = np.pad(img, pad_width, mode='reflect')

            for i in range(cuts):
                for j in range(cuts):
                    e, f = i * crop_x, (i + 1) * crop_x + 2 * win_x
                    g, h = j * crop_y, (j + 1) * crop_y + 2 * win_y
                    if img.ndim >= 4:
                        data = padded_img[:, e:f, g:h, :]
                    else:
                        data = padded_img[e:f, g:h, :]

                    predicted = self.tf_client.post_image(data, model_name, model_version)
                    if tf_results is None:
                        tf_results = np.zeros(list(img.shape)[:-1] + [predicted.shape[-1]])
                        self.logger.debug('initialized output tensor of shape %s', tf_results.shape)

                    a, b = i * crop_x, (i + 1) * crop_x
                    c, d = j * crop_y, (j + 1) * crop_y

                    if predicted.ndim >= 4:
                        tf_results[:, a:b, c:d, :] = predicted[:, win_x:-win_x, win_y:-win_y, :]
                    else:
                        tf_results[a:b, c:d, :] = predicted[win_x:-win_x, win_y:-win_y, :]
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

    def _consume(self):
        # process each unprocessed hash
        for redis_hash in self.iter_redis_hashes():
            hash_values = self.redis.hgetall(redis_hash)
            self.logger.debug('Found hash to process "%s": %s',
                redis_hash, json.dumps(hash_values, indent=4))

            self.redis.hset(redis_hash, 'processed', 'processing')
            self.logger.debug('processing image: %s', redis_hash)

            try:
                new_image_path = self.process_image(
                    hash_values.get('file_name'),
                    hash_values.get('url'),
                    hash_values.get('model_name'),
                    hash_values.get('model_version'),
                    hash_values.get('cuts'))

                self.redis.hmset(redis_hash, {
                    'output_url': new_image_path,
                    'processed': 'yes'
                })
            except Exception as err:
                self.logger.error('Failed to process redis key %s. Error: %s',
                    redis_hash, err)

    def consume(self, interval):
        if not str(interval).isdigit():
            raise ValueError('Expected `interval` to be a number. '
                             'Got {}'.format(type(interval)))

        while True:
            try:
                self._consume()
            except Exception as err:
                self.logger.error(err)

            sleep(interval)
