# Copyright 2016-2018 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/kiosk-data-processing/LICENSE
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
"""Tests for post-processing functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from skimage.external import tifffile as tiff

from redis_consumer import consumers
from redis_consumer import utils


def _get_image(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h) * 64
    variance = np.random.rand(img_w, img_h) * (255 - 64)
    img = np.random.rand(img_w, img_h) * variance + bias
    return img


class DummyRedis(object):
    def __init__(self, prefix='predict', status='new'):
        self.prefix = '/'.join(x for x in prefix.split('/') if x)
        self.status = status

    def keys(self):
        return [
            '{}_{}_{}'.format(self.prefix, self.status, 'x.tiff'),
            '{}_{}_{}'.format(self.prefix, 'other', 'x.zip'),
            '{}_{}_{}'.format('other', self.status, 'x.TIFF'),
            '{}_{}_{}'.format(self.prefix, self.status, 'x.ZIP'),
            '{}_{}_{}'.format(self.prefix, 'other', 'x.tiff'),
            '{}_{}_{}'.format('other', self.status, 'x.zip'),
        ]

    def expected_keys(self, suffix=None):
        for k in self.keys():
            v = k.split('_')
            if v[0] == self.prefix:
                if v[1] == self.status:
                    if suffix:
                        if v[-1].lower().endswith(suffix):
                            yield k
                    else:
                        yield k

    def hmset(self, rhash, hvals):  # pylint: disable=W0613
        return hvals

    def hget(self, rhash, field):
        if field == 'status':
            return rhash.split('_')[1]
        elif field == 'file_name':
            return rhash.split('_')[-1]
        elif field == 'input_file_name':
            return rhash.split('_')[-1]
        elif field == 'output_file_name':
            return rhash.split('_')[-1]
        return False

    def hset(self, rhash, status, value):  # pylint: disable=W0613
        return {status: value}

    def hgetall(self, rhash):  # pylint: disable=W0613
        return {
            'model_name': 'model',
            'model_version': '0',
            'field': '61',
            'cuts': '0',
            'postprocess_function': '',
            'preprocess_function': '',
            'file_name': rhash.split('_')[-1],
            'input_file_name': rhash.split('_')[-1],
            'output_file_name': rhash.split('_')[-1]
        }

    def type(self, key):  # pylint: disable=W0613
        return 'hash'


class DummyStorage(object):
    def __init__(self, num=3):
        self.num = num

    def download(self, path, dest):
        if path.lower().endswith('.zip'):
            paths = []
            for i in range(self.num):
                img = _get_image()
                base, ext = os.path.splitext(path)
                _path = '{}{}{}'.format(base, i, ext)
                tiff.imsave(os.path.join(dest, _path), img)
                paths.append(_path)
            return utils.zip_files(paths, dest)
        img = _get_image()
        tiff.imsave(os.path.join(dest, path), img)
        return path

    def upload(self, zip_path, subdir=None):  # pylint: disable=W0613
        return True, True

    def get_public_url(self, zip_path):  # pylint: disable=W0613
        return True


class TestConsumer(object):

    def test_iter_redis_hashes(self):
        prefix = 'prefix'
        status = 'new'
        redis = DummyRedis(prefix, status)
        consumer = consumers.Consumer(redis, None)
        keys = [k for k in consumer.iter_redis_hashes(status, prefix)]
        # test filter by prefix and status
        assert keys == [k for k in redis.expected_keys()]
        # test no status check
        keys = [k for k in consumer.iter_redis_hashes(None, prefix)]
        expected = [k for k in redis.keys() if k.startswith(prefix)]
        np.testing.assert_array_equal(keys, expected)

    def test_handle_error(self):
        global _redis_values
        _redis_values = None

        class _DummyRedis(object):
            def hmset(self, hash, hvals):
                global _redis_values
                _redis_values = hvals

        redis = _DummyRedis()
        consumer = consumers.Consumer(redis, None)
        err = Exception('test exception')
        consumer._handle_error(err, 'redis-hash')
        assert isinstance(_redis_values, dict)
        assert 'status' in _redis_values and 'reason' in _redis_values
        assert _redis_values.get('status') == 'failed'

    def test_consume(self):
        N = 4
        status = 'new'
        prefix = 'prefix'
        consumer = consumers.Consumer(DummyRedis(), DummyStorage())
        mock_hashes = lambda s, p: ['{}/{}'.format(p, i) for i in range(N)]
        consumer.iter_redis_hashes = mock_hashes
        # test that _consume is called on each hash
        global _processed
        _processed = 0

        def F(x):
            global _processed
            _processed += 1

        consumer._consume = F
        consumer.consume(status, prefix)
        assert _processed == N
        # error inside _consume does not raise
        consumer._consume = lambda x: 1 / 0
        consumer.consume(status, prefix)

    def test__consume(self):
        with np.testing.assert_raises(NotImplementedError):
            consumer = consumers.Consumer(None, None)
            consumer._consume('hash')


class TestImageFileConsumer(object):

    def test_process_big_image(self):
        name = 'model'
        version = 0
        field = 11
        cuts = 2

        img = np.expand_dims(_get_image(300, 300), axis=-1)
        img = np.expand_dims(img, axis=0)

        redis = None
        storage = None
        consumer = consumers.ImageFileConsumer(redis, storage)

        consumer.grpc_image = lambda x, y, z: x
        res = consumer.process_big_image(cuts, img, field, name, version)
        np.testing.assert_equal(res, img)

    def test_iter_redis_hashes(self):
        prefix = 'prefix'
        status = 'new'
        redis = DummyRedis(prefix, status)
        consumer = consumers.ImageFileConsumer(redis, None)
        keys = [k for k in consumer.iter_redis_hashes(status, prefix)]
        # test filter by prefix and status
        expected = list(consumer.redis.expected_keys(suffix='tiff'))
        np.testing.assert_array_equal(keys, expected)

    def test__consume(self):
        prefix = 'prefix'
        status = 'new'
        redis = DummyRedis(prefix, status)
        storage = DummyStorage()
        consumer = consumers.ImageFileConsumer(redis, storage)

        def _handle_error(err, rhash):  # pylint: disable=W0613
            raise err

        def grpc_image_multi(data, *args, **kwargs):  # pylint: disable=W0613
            return np.array(tuple(list(data.shape) + [2]))

        dummyhash = '{}_test.tiff'.format(prefix)

        # consumer._handle_error = _handle_error
        consumer.grpc_image = grpc_image_multi
        consumer._consume(dummyhash)

        # test mutli-channel
        def grpc_image(data, *args, **kwargs):  # pylint: disable=W0613
            return data

        # test with cuts > 0
        redis.hgetall = lambda x: {
            'model_name': 'model',
            'model_version': '0',
            'field': '61',
            'cuts': '2',
            'postprocess_function': '',
            'preprocess_function': '',
            'file_name': 'test_image.tiff',
            'input_file_name': 'test_image.tiff',
            'output_file_name': 'test_image.tiff'
        }
        consumer = consumers.ImageFileConsumer(redis, storage)
        consumer._handle_error = _handle_error
        consumer.grpc_image = grpc_image
        consumer._consume(dummyhash)


class TestZipFileConsumer(object):

    def test_iter_redis_hashes(self):
        prefix = 'prefix'
        status = 'new'
        redis = DummyRedis(prefix, status)
        consumer = consumers.ZipFileConsumer(redis, None)
        print(redis.keys())
        keys = [k for k in consumer.iter_redis_hashes(status, prefix)]
        # test filter by prefix and status
        expected = list(consumer.redis.expected_keys(suffix='.zip'))
        assert keys == [k for k in expected]

    def test___upload_archived_images(self):
        N = 3
        prefix = 'prefix'
        status = 'new'
        redis = DummyRedis(prefix, status)
        storage = DummyStorage(num=N)
        consumer = consumers.ZipFileConsumer(redis, storage)
        hsh = consumer._upload_archived_images({'input_file_name': 'test.zip'})
        assert len(hsh) == N

    def test__consume(self):
        N = 3
        prefix = 'prefix'
        status = 'new'
        _redis = DummyRedis(prefix, status)
        redis = DummyRedis(prefix, status)
        storage = DummyStorage(num=N)

        # test `status` = "done"
        hget = lambda h, k: 'done' if k == 'status' else _redis.hget(h, k)
        redis.hget = hget
        consumer = consumers.ZipFileConsumer(redis, storage)
        dummyhash = '{}_test.zip'.format(prefix)
        consumer._consume(dummyhash)

        # test `status` = "failed"
        hget = lambda h, k: 'failed' if k == 'status' else _redis.hget(h, k)
        redis.hget = hget
        consumer = consumers.ZipFileConsumer(redis, storage)
        dummyhash = '{}_test.zip'.format(prefix)
        consumer._consume(dummyhash)

        # test mixed `status` = "waiting" and "done"
        global counter
        counter = 0

        def hget_wait(h, k):
            if k == 'status':
                global counter
                status = 'waiting' if counter % 2 == 0 else 'done'
                counter += 1
                return status
            else:
                return _redis.hget(h, k)

        redis.hget = hget_wait
        consumer = consumers.ZipFileConsumer(redis, storage)
        dummyhash = '{}_test.zip'.format(prefix)
        consumer._consume(dummyhash)
