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
            self.logger.debug('Redis Keys: %s', keys)
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

    def process_image(self, img_name, img_url, model_name, version):
        """POSTS image data to tf-serving then saves the result
        as a zip and uploads into the cloud bucket.
        # Arguments:
            img_name: path to cloud destination of image file
            model_name: name of model in tf-serving
            version: integer version number of model in tf-serving
        # Returns:
            output_url: URL of results zip file
        """
        downloaded_image_path = self.storage.download(img_name, img_url)

        self.logger.debug('Loading the image into numpy array')
        img = img_to_array(Image.open(downloaded_image_path))
        self.logger.debug('Loaded image into numpy array with '
                          'shape %s', img.shape)

        # Get tf-serving predictions of image
        tf_results = self.tf_client.post_image(img, model_name, version)

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
        for h in self.iter_redis_hashes():
            hash_values = self.redis.hgetall(h)
            self.logger.debug('Found hash to process "%s": %s',
                h, json.dumps(hash_values, indent=4))

            self.redis.hset(h, 'processed', 'processing')
            self.logger.debug('processing image: %s', h)

            new_image_path = self.process_image(
                h,
                hash_values['url'],
                hash_values['model_name'],
                hash_values['model_version'])

            self.redis.hmset(h, {
                'output_url': new_image_path,
                'processed': 'yes'
            })

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
