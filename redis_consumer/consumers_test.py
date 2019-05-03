# Copyright 2016-2019 The Van Valen Lab at the California Institute of
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
import copy

import redis
import numpy as np
from skimage.external import tifffile as tiff

import pytest

from redis_consumer import consumers
from redis_consumer import utils


def _get_image(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h) * 64
    variance = np.random.rand(img_w, img_h) * (255 - 64)
    img = np.random.rand(img_w, img_h) * variance + bias
    return img


class DummyRedis(object):
    def __init__(self, items=[], prefix='predict', status='new'):
        self.work_queue = copy.copy(items)
        self.processing_queue = []
        self.prefix = '/'.join(x for x in prefix.split('/') if x)
        self.status = status
        self.keys = [
            '{}_{}_{}'.format(self.prefix, self.status, 'x.tiff'),
            '{}_{}_{}'.format(self.prefix, 'other', 'x.zip'),
            '{}_{}_{}'.format('other', self.status, 'x.TIFF'),
            '{}_{}_{}'.format(self.prefix, self.status, 'x.ZIP'),
            '{}_{}_{}'.format(self.prefix, 'other', 'x.tiff'),
            '{}_{}_{}'.format('other', self.status, 'x.zip'),
        ]

    def rpoplpush(self, src, dst):
        try:
            x = self.work_queue.pop()
            self.processing_queue.insert(0, x)
            return x
        except IndexError:
            return None

    def lpush(self, name, *values):
        self.work_queue = list(values) + self.work_queue
        return len(values)

    def lrem(self, name, count, value):
        self.processing_queue.remove(value)

    def scan_iter(self, match=None, count=None):
        if match:
            return (k for k in self.keys if k.startswith(match[:-1]))
        return (k for k in self.keys)

    def expected_keys(self, suffix=None):
        for k in self.keys:
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

    def expire(self, name, time):  # pylint: disable=W0613
        return 1

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

    def test_get_redis_hash(self):
        # test emtpy queue
        items = []
        redis_client = DummyRedis(items)
        consumer = consumers.Consumer(redis_client, None, 'q')
        rhash = consumer.get_redis_hash()
        assert rhash is None

        # test some valid/invalid items
        items = ['item%s' % x for x in range(1, 4)]
        redis_client = DummyRedis(items)

        consumer = consumers.Consumer(redis_client, None, 'q')
        consumer.is_valid_hash = lambda x: x == 'item1'

        rhash = consumer.get_redis_hash()

        assert rhash == items[0]
        assert redis_client.work_queue == items[1:]
        assert redis_client.processing_queue == items[0:1]

    def test_update_status(self):
        global _redis_values
        _redis_values = None

        class _DummyRedis(object):
            def hmset(self, _, hvals):
                global _redis_values
                _redis_values = hvals

        consumer = consumers.Consumer(_DummyRedis(), None, 'q')
        status = 'updated_status'
        consumer.update_status('redis-hash', status, {
            'new_field': True
        })
        assert isinstance(_redis_values, dict)
        assert 'status' in _redis_values and 'new_field' in _redis_values
        assert _redis_values.get('status') == status
        assert _redis_values.get('new_field') is True

        with pytest.raises(ValueError):
            consumer.update_status('redis-hash', status, 'data')

    def test_handle_error(self):
        global _redis_values
        _redis_values = None

        class _DummyRedis(object):
            def hmset(self, _, hvals):
                global _redis_values
                _redis_values = hvals

        consumer = consumers.Consumer(_DummyRedis(), None, 'q')
        err = Exception('test exception')
        consumer._handle_error(err, 'redis-hash')
        assert isinstance(_redis_values, dict)
        assert 'status' in _redis_values and 'reason' in _redis_values
        assert _redis_values.get('status') == 'failed'

    def test_consume(self):
        items = ['item%s' % x for x in range(1, 4)]
        N = 1  # using a queue, only one key is processed per consume()
        consumer = consumers.Consumer(DummyRedis(items), DummyStorage(), 'q')

        # test that _consume is called on each hash
        global _processed
        _processed = 0

        def F(*_):
            global _processed
            _processed += 1

        consumer._consume = F
        consumer.consume()
        assert _processed == N

        # error inside _consume calls _handle_error
        consumer._consume = lambda x: 1 / 0
        consumer._handle_error = F
        consumer.consume()
        assert _processed == N + 1

    def test__consume(self):
        with np.testing.assert_raises(NotImplementedError):
            consumer = consumers.Consumer(None, None, 'q')
            consumer._consume('hash')


class TestImageFileConsumer(object):

    def test_is_valid_hash(self):
        items = ['item%s' % x for x in range(1, 4)]

        storage = DummyStorage()
        redis_client = DummyRedis(items)
        redis_client.hget = lambda *x: x[0]

        consumer = consumers.ImageFileConsumer(redis_client, storage, 'q')
        assert consumer.is_valid_hash(None) is False
        assert consumer.is_valid_hash('file.ZIp') is False
        assert consumer.is_valid_hash('Anything else') is True

    def test_process_big_image(self):
        name = 'model'
        version = 0
        field = 11
        cuts = 2

        img = np.expand_dims(_get_image(300, 300), axis=-1)
        img = np.expand_dims(img, axis=0)

        redis_client = None
        storage = None
        consumer = consumers.ImageFileConsumer(redis_client, storage, 'q')

        # image should be chopped into cuts**2 pieces and reassembled
        consumer.grpc_image = lambda x, y, z: x
        res = consumer.process_big_image(cuts, img, field, name, version)
        np.testing.assert_equal(res, img)

    def test__consume(self):
        prefix = 'prefix'
        status = 'new'
        redis_client = DummyRedis(prefix, status)
        storage = DummyStorage()
        consumer = consumers.ImageFileConsumer(redis_client, storage, 'q')

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
        redis.hmset = lambda x, y: True
        consumer = consumers.ImageFileConsumer(redis, storage, 'q')
        consumer._handle_error = _handle_error
        consumer.grpc_image = grpc_image
        consumer._consume(dummyhash)


class TestZipFileConsumer(object):

    def test_is_valid_hash(self):
        items = ['item%s' % x for x in range(1, 4)]

        storage = DummyStorage()
        redis_client = DummyRedis(items)
        redis_client.hget = lambda *x: x[0]

        consumer = consumers.ZipFileConsumer(redis_client, storage, 'q')
        assert consumer.is_valid_hash(None) is False
        assert consumer.is_valid_hash('file.ZIp') is True
        assert consumer.is_valid_hash('Anything else') is False

    def test__upload_archived_images(self):
        N = 3
        items = ['item%s' % x for x in range(1, 4)]
        redis_client = DummyRedis(items)
        storage = DummyStorage(num=N)
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'q')
        hsh = consumer._upload_archived_images({'input_file_name': 'test.zip'})
        assert len(hsh) == N

    def test__consume(self):
        N = 3
        prefix = 'predict'
        items = ['item%s' % x for x in range(1, 4)]
        _redis = DummyRedis(items)
        redis_client = DummyRedis(items)
        storage = DummyStorage(num=N)

        # test `status` = "done"
        hget = lambda h, k: 'done' if k == 'status' else _redis.hget(h, k)
        redis_client.hget = hget
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'q')
        dummyhash = '{}_test.zip'.format(prefix)
        consumer._consume(dummyhash)

        # test `status` = "failed"
        hget = lambda h, k: 'failed' if k == 'status' else _redis.hget(h, k)
        redis_client.hget = hget
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'q')
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
            return _redis.hget(h, k)

        redis_client.hget = hget_wait
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'q')
        dummyhash = '{}_test.zip'.format(prefix)
        consumer._consume(dummyhash)
