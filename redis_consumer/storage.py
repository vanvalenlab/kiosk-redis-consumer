# Copyright 2016-2021 The Van Valen Lab at the California Institute of
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
"""Storage Interface to upload / download files from / to the cloud"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import random
import socket
import time
import timeit
import urllib3

import boto3
import google.auth.exceptions
import google.cloud.exceptions
import google.cloud.storage
import requests

from redis_consumer import settings


class StorageException(Exception):
    """Custom Exception for the Storage classes"""
    pass


def get_client(bucket):
    """Get the Storage Client appropriate for the bucket.

    Args:
        bucket (str): Bucket including

    Returns:
        ~Storage: Client for interacting with the cloud.
    """
    try:
        protocol, bucket_name = str(bucket).lower().split('://', 1)
    except ValueError:
        raise ValueError('Invalid storage bucket name: {}'.format(bucket))

    logger = logging.getLogger('storage.get_client')
    if protocol == 's3':
        storage_client = S3Storage(bucket_name)
    elif protocol == 'gs':
        storage_client = GoogleStorage(bucket_name)
    else:
        errmsg = 'Unknown STORAGE_BUCKET protocol: %s'
        logger.error(errmsg, protocol)
        raise ValueError(errmsg % protocol)
    return storage_client


class Storage(object):
    """General class to interact with cloud storage buckets.
    Supported cloud stroage provider will have child class implementations.

    Args:
        bucket: cloud storage bucket name
        download_dir: path to local directory to save downloaded files
    """

    def __init__(self, bucket,
                 download_dir=settings.DOWNLOAD_DIR,
                 max_backoff=settings.STORAGE_MAX_BACKOFF):
        self.bucket = bucket
        self.download_dir = download_dir
        self.output_dir = 'output'
        self.logger = logging.getLogger(str(self.__class__.__name__))
        self.max_backoff = max_backoff

        # try to write the download dir in case it does not exist.
        try:
            os.mkdir(self.download_dir)
        except OSError:
            pass

    def get_backoff(self, attempts):
        """Get backoff time based on previous number of attempts"""
        milis = random.randint(1, 1000) / 1000
        exponential = 2 ** attempts + milis
        backoff = min(exponential, self.max_backoff)
        return backoff

    def get_storage_client(self):
        """Returns the storage API client"""
        raise NotImplementedError

    def get_download_path(self, filepath, download_dir=None):
        """Get local filepath for soon-to-be downloaded file.

        Args:
            filepath: key of file in cloud storage to download
            download_dir: path to directory to save file

        Returns:
            dest: local path to downloaded file
        """
        if download_dir is None:
            download_dir = self.download_dir
        no_upload_dir = os.path.join(*(filepath.split(os.path.sep)[1:]))
        dest = os.path.join(download_dir, no_upload_dir)
        if not os.path.isdir(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        return dest

    def download(self, filepath, download_dir):
        """Download a  file from the cloud storage bucket.

        Args:
            filepath: key of file in cloud storage to download
            download_dir: path to directory to save file

        Returns:
            dest: local path to downloaded file
        """
        raise NotImplementedError

    def upload(self, filepath, subdir=None):
        """Upload a file to the cloud storage bucket.

        Args:
            filepath: local path to file to upload

        Returns:
            dest: key of uploaded file in cloud storage
        """
        raise NotImplementedError


class GoogleStorage(Storage):
    """Interact with Google Cloud Storage buckets.

    Args:
        bucket: cloud storage bucket name
        download_dir: path to local directory to save downloaded files
    """

    def __init__(self, bucket,
                 download_dir=settings.DOWNLOAD_DIR,
                 max_backoff=settings.STORAGE_MAX_BACKOFF):
        super(GoogleStorage, self).__init__(bucket, download_dir, max_backoff)
        self.bucket_url = 'www.googleapis.com/storage/v1/b/{}/o'.format(bucket)
        self._network_errors = (
            socket.gaierror,
            google.cloud.exceptions.TooManyRequests,
            google.cloud.exceptions.InternalServerError,
            google.cloud.exceptions.ServiceUnavailable,
            google.cloud.exceptions.GatewayTimeout,
            urllib3.exceptions.MaxRetryError,
            urllib3.exceptions.NewConnectionError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            google.auth.exceptions.RefreshError,
            google.auth.exceptions.TransportError,
        )

    def get_storage_client(self):
        """Returns the storage API client"""
        attempts = 0
        while True:
            try:
                return google.cloud.storage.Client()
            except OSError as err:
                if attempts < 3:
                    backoff = self.get_backoff(attempts)
                    attempts += 1
                    self.logger.warning('Encountered error while creating '
                                        'storage client: %s', err)
                    time.sleep(backoff)
                else:
                    raise err

    def get_public_url(self, filepath):
        """Get the public URL to download the file.

        Args:
            filepath: key to file in cloud storage

        Returns:
            url: Public URL to download the file
        """
        retrying = True
        attempts = 0
        while retrying:
            try:
                client = self.get_storage_client()
                bucket = client.get_bucket(self.bucket)
                blob = bucket.blob(filepath)
                blob.make_public()
                retrying = False
                return blob.public_url

            except self._network_errors as err:
                backoff = self.get_backoff(attempts)
                self.logger.warning('Encountered %s: %s.  Backing off for %s '
                                    'seconds...', type(err).__name__, err,
                                    backoff)
                time.sleep(backoff)
                attempts += 1
                retrying = True  # Unneccessary but explicit

            except Exception as err:
                retrying = False
                self.logger.error('Encountered %s: %s during make_public %s.',
                                  type(err).__name__, err, filepath)
                raise err

    def upload(self, filepath, subdir=None):
        """Upload a file to the cloud storage bucket.

        Args:
            filepath: local path to file to upload

        Returns:
            dest: key of uploaded file in cloud storage
        """
        start = timeit.default_timer()
        self.logger.debug('Uploading %s to bucket %s.', filepath, self.bucket)
        retrying = True
        attempts = 0
        while retrying:
            client = self.get_storage_client()
            try:
                dest = os.path.basename(filepath)
                if subdir:
                    if str(subdir).startswith('/'):
                        subdir = subdir[1:]
                    dest = os.path.join(subdir, dest)
                dest = os.path.join(self.output_dir, dest)
                bucket = client.get_bucket(self.bucket)
                blob = bucket.blob(dest)
                blob.upload_from_filename(filepath, predefined_acl='publicRead')
                self.logger.debug('Uploaded %s to bucket %s in %s seconds.',
                                  filepath, self.bucket,
                                  timeit.default_timer() - start)
                retrying = False
                return dest, blob.public_url
            except self._network_errors as err:
                backoff = self.get_backoff(attempts)
                self.logger.warning('Encountered %s: %s.  Backing off for %s '
                                    'seconds...', type(err).__name__, err,
                                    backoff)
                time.sleep(backoff)
                attempts += 1
                retrying = True  # Unneccessary but explicit

            except Exception as err:
                retrying = False
                self.logger.error('Encountered %s: %s while uploading %s.',
                                  type(err).__name__, err, filepath)
                raise err

    def download(self, filepath, download_dir=None):
        """Download a  file from the cloud storage bucket.

        Args:
            filepath: key of file in cloud storage to download
            download_dir: path to directory to save file

        Returns:
            dest: local path to downloaded file
        """
        dest = self.get_download_path(filepath, download_dir)
        self.logger.debug('Downloading %s to %s.', filepath, dest)
        retrying = True
        attempts = 0
        while retrying:
            client = self.get_storage_client()
            try:
                start = timeit.default_timer()
                blob = client.get_bucket(self.bucket).blob(filepath)
                blob.download_to_filename(dest)
                self.logger.debug('Downloaded %s from bucket %s in %s seconds.',
                                  dest, self.bucket,
                                  timeit.default_timer() - start)
                return dest

            except self._network_errors as err:
                backoff = self.get_backoff(attempts)
                self.logger.warning('Encountered %s: %s.  Backing off for %s '
                                    'seconds and...', type(err).__name__, err,
                                    backoff)
                time.sleep(backoff)
                attempts += 1
                retrying = True  # Unneccessary but explicit

            except Exception as err:
                retrying = False
                self.logger.error('Encountered %s: %s while downloading %s.',
                                  type(err).__name__, err, filepath)
                raise err


class S3Storage(Storage):
    """Interact with Amazon S3 buckets.

    Args:
        bucket: cloud storage bucket name
        download_dir: path to local directory to save downloaded files
    """

    def __init__(self, bucket,
                 download_dir=settings.DOWNLOAD_DIR,
                 max_backoff=settings.STORAGE_MAX_BACKOFF):
        super(S3Storage, self).__init__(bucket, download_dir, max_backoff)
        self.bucket_url = 's3.amazonaws.com/{}'.format(bucket)

    def get_storage_client(self):
        """Returns the storage API client"""
        return boto3.client(
            's3',
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)

    def get_public_url(self, filepath):
        """Get the public URL to download the file.

        Args:
            filepath: key to file in cloud storage

        Returns:
            url: Public URL to download the file
        """
        return 'https://{url}/{obj}'.format(url=self.bucket_url, obj=filepath)

    def upload(self, filepath, subdir=None):
        """Upload a file to the cloud storage bucket.

        Args:
            filepath: local path to file to upload

        Returns:
            dest: key of uploaded file in cloud storage
        """
        start = timeit.default_timer()
        client = self.get_storage_client()
        dest = os.path.basename(filepath)
        if subdir:
            if str(subdir).startswith('/'):
                subdir = subdir[1:]
            dest = os.path.join(subdir, dest)
        dest = os.path.join(self.output_dir, dest)
        self.logger.debug('Uploading %s to bucket %s.', filepath, self.bucket)
        try:
            client.upload_file(filepath, self.bucket, dest)
            self.logger.debug('Uploaded %s to bucket %s in %s seconds.',
                              filepath, self.bucket,
                              timeit.default_timer() - start)
            return dest, self.get_public_url(dest)
        except Exception as err:
            self.logger.error('Encountered %s: %s while uploading %s.',
                              type(err).__name__, err, filepath)
            raise err

    def download(self, filepath, download_dir=None):
        """Download a  file from the cloud storage bucket.

        Args:
            filepath: key of file in cloud storage to download
            download_dir: path to directory to save file

        Returns:
            dest: local path to downloaded file
        """
        start = timeit.default_timer()
        client = self.get_storage_client()
        # Bucket keys shouldn't start with "/"
        if filepath.startswith('/'):
            filepath = filepath[1:]

        dest = self.get_download_path(filepath, download_dir)
        self.logger.debug('Downloading %s to %s.', filepath, dest)
        try:
            client.download_file(self.bucket, filepath, dest)
            self.logger.debug('Downloaded %s from bucket %s in %s seconds.',
                              dest, self.bucket, timeit.default_timer() - start)
            return dest
        except Exception as err:
            self.logger.error('Encountered %s: %s while downloading %s.',
                              type(err).__name__, err, filepath)
            raise err
