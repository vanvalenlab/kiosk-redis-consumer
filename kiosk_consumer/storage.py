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
"""Storage Interface to upload / download files
from a variety of cloud platforms.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import boto3
from google.cloud import storage as google_storage
from PIL import Image

from kiosk_consumer import settings


class Storage(object):

    def __init__(self, bucket):
        self._client = None
        self.bucket_url = None
        self.bucket = bucket
        self.download_dir = settings.DOWNLOAD_DIR
        self.logger = logging.getLogger(str(self.__class__.__name__))

    def _get_public_url(self, filepath):
        return 'https://{url}/{obj}'.format(url=self.bucket_url, obj=filepath)

    def get_download_path(self, filename):
        no_upload_dir = os.path.join(*(filename.split(os.path.sep)[1:]))
        return os.path.join(self.download_dir, no_upload_dir)

    def download(self, filename, url):
        raise NotImplementedError

    def upload(self, file_path):
        raise NotImplementedError


class GoogleStorage(Storage):

    def __init__(self, bucket):
        super(GoogleStorage, self).__init__(bucket)
        self._client = google_storage.Client()
        self.bucket_url = 'www.googleapis.com/storage/v1/b/{}/o'.format(bucket)

    def upload(self, filepath):
        """Upload a file to the cloud storage bucket"""
        try:
            self.logger.debug('Uploading %s to bucket %s.',
                filepath, self.bucket)
            dest = os.path.basename(filepath)
            bucket = self._client.get_bucket(self.bucket)
            blob = bucket.blob(dest)
            blob.upload_from_filename(filepath)
            blob.make_public()
            self.logger.debug('Successfully uploaded {} to bucket {}'.format(
                filepath, self.bucket))
            return blob.public_url
        except Exception as err:
            self.logger.error('Error while uploading image {}: {}'.format(
                filepath, err))
            raise err

    def download(self, filename, url):
        """Download a  file from the cloud storage bucket"""
        dest = self.get_download_path(filename)
        self.logger.debug('Downloading %s to %s.', url, dest)
        try:
            blob = self._client.get_bucket(self.bucket).blob(filename)
            with open(dest, 'wb') as new_file:
                blob.download_to_file(new_file)
            return dest
        except Exception as err:
            self.logger.error('Error while downloading image {}: {}'.format(
                url, err))
            raise err


class S3Storage(Storage):

    def __init__(self, bucket):
        super(S3Storage, self).__init__(bucket)
        self._client = boto3.client(
            's3',
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)
        self.bucket_url = 's3.amazonaws.com/{}'.format(bucket)

    def upload(self, filepath):
        """Upload a file to the cloud storage bucket"""
        dest = os.path.basename(filepath)
        self.logger.debug('Uploading %s to bucket %s.', filepath, self.bucket)
        try:
            self._client.upload_file(filepath, self.bucket, dest)
            self.logger.debug('Successfully uploaded {} to bucket {}'.format(
                filepath, self.bucket))
            return self._get_public_url(dest)
        except Exception as err:
            self.logger.error('Error while uploading image {}: {}'.format(
                filepath, err))
            raise err
    
    def download(self, filename, url):
        """Download a  file from the cloud storage bucket"""
        dest = self.get_download_path(filename)
        self.logger.debug('Downloading %s to %s.', url, dest)
        try:
            self._client.download_file(self.bucket, filename, dest)
            self.logger.debug('downloaded %s', dest)
            return dest
        except Exception as err:
            self.logger.error('Error while downloading image {}: {}'.format(
                url, err))
            raise err
