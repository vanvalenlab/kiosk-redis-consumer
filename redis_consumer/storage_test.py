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
"""Tests for API Storage classes"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from google.cloud.exceptions import TooManyRequests

import pytest

from redis_consumer import storage
from redis_consumer import utils


def throw_critical_error(*_, **__):
    raise OSError('Thrown on purpose')


def fast(*_, **__):
    return 0


class _Singleton(type):
    """A metaclass that creates a Singleton base class when called.

    https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """
    _instances = {}

    def __call__(cls, *_, **__):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*_, **__)
        return cls._instances[cls]


class Singleton(_Singleton('SingletonMeta', (object,), {})):
    pass


class DummyGoogleClient(Singleton):
    public_url = 'public-url'

    def __call__(self, *args, **kwargs):
        return self

    def __init__(self, *_, **__):
        self.storage_error = True

    def __getattr__(self, a):
        return self

    def _throw_first_error(self):
        if self.storage_error:
            self.storage_error = False
            raise TooManyRequests('thrown-on-purpose')
        self.storage_error = True
        return self

    def make_public(self):
        self._throw_first_error()
        return self

    def upload_from_filename(self, dest, **_):
        self._throw_first_error()
        assert os.path.exists(dest)

    def download_to_filename(self, dest, **_):
        self._throw_first_error()
        assert dest.endswith('/test/file.txt')


class DummyS3Client(object):

    def __init__(self, *_, **__):
        pass

    def download_file(self, bucket, path, dest, **_):
        assert path.startswith('test')

    def upload_file(self, path, bucket, dest, **_):
        assert os.path.exists(path)


def test_get_client():
    aws = storage.get_client('s3://bucket')
    AWS = storage.get_client('S3://anotherbucket')
    assert isinstance(aws, type(AWS))

    # TODO: set GCLOUD env vars to test this
    # with pytest.raises(OSError):
    gke = storage.get_client('gs://bucket')
    GKE = storage.get_client('GS://anotherbucket')
    assert isinstance(gke, type(GKE))

    bad_values = ['s3', 'gs', 's3:/badval', 'gs//badval']
    for bad_value in bad_values:
        with pytest.raises(ValueError):
            _ = storage.get_client(bad_value)


class TestStorage(object):

    def test_get_backoff(self):
        max_backoff = 30
        client = storage.Storage('bucket', max_backoff=max_backoff)
        backoff = client.get_backoff(attempts=0)
        assert 1 < backoff < 2

        backoff = client.get_backoff(attempts=3)
        assert 8 < backoff < 9

        backoff = client.get_backoff(attempts=5)
        assert backoff == max_backoff

    def test_get_download_path(self, mocker, tmpdir):
        tmpdir = str(tmpdir)
        mocker.patch('redis_consumer.storage.Storage.get_storage_client',
                     lambda *x: True)

        bucket = 'test-bucket'
        stg = storage.Storage(bucket, tmpdir)
        filekey = 'upload_dir/key/to.zip'
        path = stg.get_download_path(filekey, tmpdir)
        path2 = stg.get_download_path(filekey)
        assert path == path2
        assert str(path).startswith(tmpdir)
        assert str(path).endswith(filekey.replace('upload_dir/', ''))


class TestGoogleStorage(object):

    def test_get_storage_client(self, tmpdir, mocker):

        mocker.patch('google.cloud.storage.Client', throw_critical_error)
        mocker.patch('redis_consumer.storage.GoogleStorage.get_backoff', fast)

        bucket = 'test-bucket'
        tmpdir = str(tmpdir)

        with pytest.raises(OSError):
            stg = storage.GoogleStorage(bucket, tmpdir)
            stg.get_storage_client()

    def test_get_public_url(self, tmpdir, mocker):
        tmpdir = str(tmpdir)
        mocker.patch('google.cloud.storage.Client', DummyGoogleClient)
        mocker.patch('redis_consumer.storage.GoogleStorage.get_backoff', fast)
        with tempfile.NamedTemporaryFile(dir=tmpdir) as temp:
            bucket = 'test-bucket'
            stg = storage.GoogleStorage(bucket, tmpdir)
            url = stg.get_public_url(temp.name)
            assert url == 'public-url'

            # test bad filename
            mocker.patch.object(DummyGoogleClient, 'make_public',
                                throw_critical_error)
            with pytest.raises(OSError):
                stg.get_public_url('file-does-not-exist')

    def test_upload(self, tmpdir, mocker):
        tmpdir = str(tmpdir)
        mocker.patch('google.cloud.storage.Client', DummyGoogleClient)
        mocker.patch('redis_consumer.storage.GoogleStorage.get_backoff', fast)
        with tempfile.NamedTemporaryFile(dir=tmpdir) as temp:
            bucket = 'test-bucket'
            stg = storage.GoogleStorage(bucket, tmpdir)

            # test succesful upload
            dest, url = stg.upload(temp.name)
            assert dest == 'output/{}'.format(os.path.basename(temp.name))

            # test succesful upload with subdir
            subdir = '/abc'
            dest, url = stg.upload(temp.name, subdir=subdir)
            assert dest == 'output{}/{}'.format(
                subdir, os.path.basename(temp.name))

            # test failed upload
            with pytest.raises(Exception):
                # self._client raises, but so does storage.upload
                dest, url = stg.upload('file-does-not-exist')

    def test_download(self, tmpdir, mocker):
        remote_file = '/test/file.txt'
        tmpdir = str(tmpdir)

        bucket = 'test-bucket'
        mocker.patch('google.cloud.storage.Client', DummyGoogleClient)
        mocker.patch('redis_consumer.storage.GoogleStorage.get_backoff', fast)

        stg = storage.GoogleStorage(bucket, tmpdir)

        # test succesful download
        dest = stg.download(remote_file, tmpdir)
        assert dest == stg.get_download_path(remote_file, tmpdir)

        # test failed download
        with pytest.raises(Exception):
            # self._client raises, but so does storage.download
            dest = stg.download('bad/file.txt', tmpdir)


class TestS3Storage(object):

    def test_get_public_url(self, tmpdir):
        bucket = 'test-bucket'
        stg = storage.S3Storage(bucket, str(tmpdir))
        url = stg.get_public_url('test')
        assert url == 'https://{}/{}'.format(stg.bucket_url, 'test')

    def test_upload(self, tmpdir, mocker):
        tmpdir = str(tmpdir)
        mocker.patch('boto3.client', DummyS3Client)
        mocker.patch('redis_consumer.storage.S3Storage.get_backoff',
                     lambda *x: 0)
        with tempfile.NamedTemporaryFile(dir=tmpdir) as temp:
            bucket = 'test-bucket'
            stg = storage.S3Storage(bucket, tmpdir)

            # test succesful upload
            dest, url = stg.upload(temp.name)
            assert dest == 'output/{}'.format(os.path.basename(temp.name))

            # test succesful upload with subdir
            subdir = '/abc'
            dest, url = stg.upload(temp.name, subdir=subdir)
            assert dest == 'output{}/{}'.format(
                subdir, os.path.basename(temp.name))

            # test failed upload
            with pytest.raises(Exception):
                # self._client raises, but so does storage.upload
                dest, url = stg.upload('file-does-not-exist')

    def test_download(self, tmpdir, mocker):
        tmpdir = str(tmpdir)
        mocker.patch('boto3.client', DummyS3Client)
        mocker.patch('redis_consumer.storage.S3Storage.get_backoff',
                     lambda *x: 0)

        remote_file = '/test/file.txt'

        bucket = 'test-bucket'
        stg = storage.S3Storage(bucket, tmpdir)

        # test succesful download
        dest = stg.download(remote_file, tmpdir)
        assert dest == stg.get_download_path(remote_file[1:], tmpdir)

        # test failed download
        with pytest.raises(Exception):
            # self._client raises, but so does storage.download
            dest = stg.download('bad/file.txt', tmpdir)
