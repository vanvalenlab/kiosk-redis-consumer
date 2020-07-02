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


# global var forcing storage clients to throw an error
global storage_error
storage_error = True

global critical_error
critical_error = False


class DummyGoogleClient(object):
    public_url = 'public-url'

    def get_bucket(self, *_, **__):
        return self

    def blob(self, *_, **__):
        return self

    def make_public(self, *_, **__):
        global storage_error
        global critical_error

        if critical_error:
            raise Exception('critical-error-thrown-on-purpose')

        if storage_error:
            storage_error = False
            raise TooManyRequests('thrown-on-purpose')
        storage_error = True
        return self

    def upload_from_filename(self, dest, **_):
        global storage_error

        if storage_error:
            storage_error = False
            raise TooManyRequests('thrown-on-purpose')
        storage_error = True
        assert os.path.exists(dest)

    def download_to_filename(self, dest, **_):
        global storage_error
        if storage_error:
            storage_error = False
            raise TooManyRequests('thrown-on-purpose')
        storage_error = True
        assert dest.endswith('/test/file.txt')


class DummyS3Client(object):
    def __init__(self, *_, **__):
        pass

    def download_file(self, bucket, path, dest, **_):
        assert path.startswith('test')

    def upload_file(self, path, bucket, dest, **_):
        assert os.path.exists(path)


def test_get_client():
    aws = storage.get_client('aws')
    AWS = storage.get_client('AWS')
    assert isinstance(aws, type(AWS))

    # TODO: set GCLOUD env vars to test this
    # with pytest.raises(OSError):
    gke = storage.get_client('gke')
    GKE = storage.get_client('GKE')
    assert isinstance(gke, type(GKE))

    with pytest.raises(ValueError):
        _ = storage.get_client('bad_value')


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

    def test_get_download_path(self, mocker):
        mocker.patch('redis_consumer.storage.Storage.get_storage_client',
                     lambda *x: True)
        with utils.get_tempdir() as tempdir:
            bucket = 'test-bucket'
            stg = storage.Storage(bucket, tempdir)
            filekey = 'upload_dir/key/to.zip'
            path = stg.get_download_path(filekey, tempdir)
            path2 = stg.get_download_path(filekey)
            assert path == path2
            assert str(path).startswith(tempdir)
            assert str(path).endswith(filekey.replace('upload_dir/', ''))


class TestGoogleStorage(object):

    def test_get_storage_client(self, tmpdir, mocker):

        def bad_google_client():
            raise OSError('thrown on purpose')

        mocker.patch('google.cloud.storage.Client', bad_google_client)
        mocker.patch('redis_consumer.storage.GoogleStorage.get_backoff',
                     lambda *x: 0)

        bucket = 'test-bucket'
        tmpdir = str(tmpdir)

        with pytest.raises(OSError):
            stg = storage.GoogleStorage(bucket, tmpdir)
            stg.get_storage_client()

    def test_get_public_url(self, tmpdir, mocker):
        tmpdir = str(tmpdir)
        mocker.patch('google.cloud.storage.Client', DummyGoogleClient)
        mocker.patch('redis_consumer.storage.GoogleStorage.get_backoff',
                     lambda *x: 0)
        with tempfile.NamedTemporaryFile(dir=tmpdir) as temp:
            bucket = 'test-bucket'
            stg = storage.GoogleStorage(bucket, tmpdir)
            url = stg.get_public_url(temp.name)
            assert url == 'public-url'

            # test bad filename
            global critical_error
            with pytest.raises(Exception):
                critical_error = True
                # client.make_public() raises error.
                stg.get_public_url('file-does-not-exist')
            critical_error = False

    def test_upload(self, tmpdir, mocker):
        tmpdir = str(tmpdir)
        mocker.patch('google.cloud.storage.Client', DummyGoogleClient)
        mocker.patch('redis_consumer.storage.GoogleStorage.get_backoff',
                     lambda *x: 0)
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
        mocker.patch('redis_consumer.storage.GoogleStorage.get_backoff',
                     lambda *x: 0)

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
