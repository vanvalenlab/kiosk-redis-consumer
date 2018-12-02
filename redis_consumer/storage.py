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
"""Storage Interface to upload / download files
from a variety of cloud platforms.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from timeit import default_timer

import boto3
from google.cloud import storage as google_storage

from redis_consumer import settings
from redis_consumer.settings import DOWNLOAD_DIR


def get_client(cloud_provider):
    """Returns the Storage Client appropriate for the cloud provider
    # Arguments:
        cloud_provider: Indicates which cloud platform (AWS vs GKE)
    # Returns:
        storage_client: Client for interacting with the cloud.
    """
    logger = logging.getLogger('storage.get_client')
    if cloud_provider == 'aws':
        storage_client = S3Storage(settings.AWS_S3_BUCKET)
    elif cloud_provider == 'gke':
        storage_client = GoogleStorage(settings.GCLOUD_STORAGE_BUCKET)
    else:
        errmsg = 'Bad value for CLOUD_PROVIDER: %s'
        logger.error(errmsg, cloud_provider)
        raise ValueError(errmsg % cloud_provider)
    return storage_client


class Storage(object):
    """General class to interact with cloud storage buckets.
    Supported cloud stroage provider will have child class implementations.
    """

    def __init__(self, bucket, download_dir=DOWNLOAD_DIR):
        self._client = None
        self.bucket = bucket
        self.download_dir = download_dir
        self.logger = logging.getLogger(str(self.__class__.__name__))

    def get_download_path(self, filename, download_dir=None):
        """Get local filepath for soon-to-be downloaded file
        # Arguments:
            filename: key of file in cloud storage to download
            download_dir: path to directory to save file
        # Returns:
            dest: local path to downloaded file
        """
        if download_dir is None:
            download_dir = self.download_dir
        no_upload_dir = os.path.join(*(filename.split(os.path.sep)[1:]))
        dest = os.path.join(download_dir, no_upload_dir)
        if not os.path.isdir(os.path.dirname(dest)):
            os.makedirs(dest)
        return dest

    def download(self, filename, download_dir):
        """Download a  file from the cloud storage bucket
        # Arguments:
            filename: key of file in cloud storage to download
            download_dir: path to directory to save file
        # Returns:
            dest: local path to downloaded file
        """
        raise NotImplementedError

    def upload(self, filepath):
        """Upload a file to the cloud storage bucket
        # Arguments:
            filepath: local path to file to upload
        # Returns:
            dest: key of uploaded file in cloud storage
        """
        raise NotImplementedError


class GoogleStorage(Storage):
    """Interact with Google Cloud Storage buckets"""

    def __init__(self, bucket, download_dir=DOWNLOAD_DIR):
        super(GoogleStorage, self).__init__(bucket, download_dir)
        self._client = google_storage.Client()
        self.bucket_url = 'www.googleapis.com/storage/v1/b/{}/o'.format(bucket)

    def get_public_url(self, filepath):
        """Get the public URL to download the file
        # Arguments:
            filepath: key to file in cloud storage
        # Returns:
            url: Public URL to download the file
        """
        bucket = self._client.get_bucket(self.bucket)
        blob = bucket.blob(filepath)
        blob.make_public()
        return blob.public_url

    def upload(self, filepath):
        """Upload a file to the cloud storage bucket
        # Arguments:
            filepath: local path to file to upload
        # Returns:
            dest: key of uploaded file in cloud storage
        """
        start = default_timer()
        self.logger.debug('Uploading %s to bucket %s.', filepath, self.bucket)
        try:
            dest = os.path.join('output', os.path.basename(filepath))
            bucket = self._client.get_bucket(self.bucket)
            blob = bucket.blob(dest)
            blob.upload_from_filename(filepath)
            self.logger.debug('Successfully uploaded %s to bucket %s in %s s',
                              filepath, self.bucket, default_timer() - start)
            return dest
        except Exception as err:
            self.logger.error('Error while uploading image %s: %s',
                              filepath, err)
            raise err

    def download(self, filename, download_dir=None):
        """Download a  file from the cloud storage bucket
        # Arguments:
            filename: key of file in cloud storage to download
            download_dir: path to directory to save file
        # Returns:
            dest: local path to downloaded file
        """
        start = default_timer()
        dest = self.get_download_path(filename, download_dir)
        self.logger.debug('Downloading %s to %s.', filename, dest)
        try:
            blob = self._client.get_bucket(self.bucket).blob(filename)
            with open(dest, 'wb') as new_file:
                blob.download_to_file(new_file)
            self.logger.debug('Downloaded %s in %s s',
                              dest, default_timer() - start)
            return dest
        except Exception as err:
            self.logger.error('Error while downloading image %s: %s',
                              filename, err)
            raise err


class S3Storage(Storage):
    """Interact with Amazon S3 buckets"""

        self._client = boto3.client(
            's3',
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)
    def __init__(self,
                 bucket,
                 download_dir=DOWNLOAD_DIR):
        super(S3Storage, self).__init__(bucket, download_dir)
        self.bucket_url = 's3.amazonaws.com/{}'.format(bucket)

    def get_public_url(self, filepath):
        """Get the public URL to download the file
        # Arguments:
            filepath: key to file in cloud storage
        # Returns:
            url: Public URL to download the file
        """
        return 'https://{url}/{obj}'.format(url=self.bucket_url, obj=filepath)

    def upload(self, filepath):
        """Upload a file to the cloud storage bucket
        # Arguments:
            filepath: local path to file to upload
        # Returns:
            dest: key of uploaded file in cloud storage
        """
        start = default_timer()
        dest = os.path.join('output', os.path.basename(filepath))
        self.logger.debug('Uploading %s to bucket %s.', filepath, self.bucket)
        try:
            self._client.upload_file(filepath, self.bucket, dest)
            self.logger.debug('Successfully uploaded %s to bucket %s in %s s',
                              filepath, self.bucket, default_timer() - start)
            return dest
        except Exception as err:
            self.logger.error('Error while uploading image %s: %s',
                              filepath, err)
            raise err

    def download(self, filename, download_dir=None):
        """Download a  file from the cloud storage bucket
        # Arguments:
            filename: key of file in cloud storage to download
            download_dir: path to directory to save file
        # Returns:
            dest: local path to downloaded file
        """
        start = default_timer()
        # Bucket keys shouldn't start with "/"
        if filename.startswith('/'):
            filename = filename[1:]

        dest = self.get_download_path(filename, download_dir)
        self.logger.debug('Downloading %s to %s.', filename, dest)
        try:
            self._client.download_file(self.bucket, filename, dest)
            self.logger.debug('Downloaded %s in %s s',
                              dest, default_timer() - start)
            return dest
        except Exception as err:
            self.logger.error('Error while downloading image %s: %s',
                              filename, err)
            raise err
