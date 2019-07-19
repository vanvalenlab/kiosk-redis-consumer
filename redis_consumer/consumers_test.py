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
import math
import random

import redis
import numpy as np
from skimage.external import tifffile as tiff

import pytest

from redis_consumer import consumers
from redis_consumer import utils
from redis_consumer import settings


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
            '{}:{}:{}'.format(self.prefix, 'x.tiff', self.status),
            '{}:{}:{}'.format(self.prefix, 'x.zip', 'other'),
            '{}:{}:{}'.format('other', 'x.TIFF', self.status),
            '{}:{}:{}'.format(self.prefix, 'x.ZIP', self.status),
            '{}:{}:{}'.format(self.prefix, 'x.tiff', 'other'),
            '{}:{}:{}'.format('other', 'x.zip', self.status),
        ]

    def rpoplpush(self, src, dst):
        if src.startswith('processing'):
            source = self.processing_queue
            dest = self.work_queue
        elif src.startswith(self.prefix):
            source = self.work_queue
            dest = self.processing_queue

        try:
            x = source.pop()
            dest.insert(0, x)
            return x
        except IndexError:
            return None

    def lpush(self, name, *values):
        self.work_queue = list(reversed(values)) + self.work_queue
        return len(self.work_queue)

    def lrem(self, name, count, value):
        self.processing_queue.remove(value)
        return count

    def llen(self, queue):
        if queue.startswith('processing'):
            return len(self.processing_queue)
        else:
            return len(self.work_queue)

    def hmget(self, rhash, *args):
        return [self.hget(rhash, a) for a in args]

    def hmset(self, rhash, hvals):  # pylint: disable=W0613
        return hvals

    def expire(self, name, time):  # pylint: disable=W0613
        return 1

    def hget(self, rhash, field):
        if field == 'status':
            return rhash.split(':')[-1]
        elif field == 'file_name':
            return rhash.split(':')[1]
        elif field == 'input_file_name':
            return rhash.split(':')[1]
        elif field == 'output_file_name':
            return rhash.split(':')[1]
        elif field == 'reason':
            return 'reason'
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
            'file_name': rhash.split(':')[1],
            'input_file_name': rhash.split(':')[1],
            'output_file_name': rhash.split(':')[1],
            'status': rhash.split(':')[-1],
            'children': 'predict:1.tiff:done,predict:2.tiff:failed,predict:3.tiff:new',
            'children:done': 'predict:4.tiff:done,predict:5.tiff:done',
            'children:failed': 'predict:6.tiff:failed,predict:7.tiff:failed',
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
        settings.EMPTY_QUEUE_TIMEOUT = 0.01  # don't sleep too long

        queue_name = 'q'
        # test emtpy queue
        items = []
        redis_client = DummyRedis(items, prefix=queue_name)
        consumer = consumers.Consumer(redis_client, None, queue_name)
        rhash = consumer.get_redis_hash()
        assert rhash is None

        # test some valid/invalid items
        items = ['item%s' % x for x in range(1, 4)]
        redis_client = DummyRedis(items, prefix=queue_name)

        consumer = consumers.Consumer(redis_client, None, queue_name)
        consumer.is_valid_hash = lambda x: x == 'item1'
        print(redis_client.work_queue)

        rhash = consumer.get_redis_hash()
        print(redis_client.work_queue)
        assert rhash == items[0]
        assert redis_client.work_queue == items[1:]
        assert redis_client.processing_queue == items[0:1]

    def test_update_key(self):
        global _redis_values
        _redis_values = None

        class _DummyRedis(object):
            def hmset(self, _, hvals):
                global _redis_values
                _redis_values = hvals

        consumer = consumers.Consumer(_DummyRedis(), None, 'q')
        status = 'updated_status'
        consumer.update_key('redis-hash', {
            'status': status,
            'new_field': True
        })
        assert isinstance(_redis_values, dict)
        assert 'status' in _redis_values and 'new_field' in _redis_values
        assert _redis_values.get('status') == status
        assert _redis_values.get('new_field') is True

        with pytest.raises(ValueError):
            consumer.update_key('redis-hash', 'data')

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
        queue_name = 'q'
        items = ['{}:{}:{}.tiff'.format(queue_name, 'new', x) for x in range(1, 4)]
        N = 1  # using a queue, only one key is processed per consume()
        consumer = consumers.Consumer(
            DummyRedis(items, prefix=queue_name), DummyStorage(), queue_name)

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

        # empty redis queue
        consumer.get_redis_hash = lambda: None
        settings.EMPTY_QUEUE_TIMEOUT = 0.1  # don't sleep too long
        consumer.consume()

        # failed and done statuses call lrem
        def lrem(key, count, value):
            global _processed
            _processed = True

        _processed = False
        redis_client = DummyRedis(items, prefix=queue_name)
        redis_client.lrem = lrem
        consumer = consumers.Consumer(redis_client, DummyStorage(), queue_name)
        consumer.get_redis_hash = lambda: '%s:f.tiff:failed' % queue_name
        print(redis_client.work_queue)
        print(redis_client.processing_queue)
        consumer.consume()
        print(redis_client.work_queue)
        print(redis_client.processing_queue)
        assert _processed is True

        _processed = False
        consumer.get_redis_hash = lambda: '{q}:f.tiff:{status}'.format(
            q=queue_name,
            status=consumer.final_status)
        consumer.consume()
        assert _processed is True

    def test__consume(self):
        with np.testing.assert_raises(NotImplementedError):
            consumer = consumers.Consumer(None, None, 'q')
            consumer._consume('predict:new:hash.tiff')


class TestImageFileConsumer(object):

    def test_is_valid_hash(self):
        items = ['item%s' % x for x in range(1, 4)]

        storage = DummyStorage()
        redis_client = DummyRedis(items)
        redis_client.hget = lambda *x: x[0]

        consumer = consumers.ImageFileConsumer(redis_client, storage, 'predict')
        assert consumer.is_valid_hash(None) is False
        assert consumer.is_valid_hash('file.ZIp') is False
        assert consumer.is_valid_hash('predict:1234567890:file.ZIp') is False
        assert consumer.is_valid_hash('track:123456789:file.zip') is False
        assert consumer.is_valid_hash('predict:123456789:file.zip') is False
        assert consumer.is_valid_hash('predict:1234567890:file.tiff') is True
        assert consumer.is_valid_hash('predict:1234567890:file.png') is True

    def test_process_big_image(self):
        name = 'model'
        version = 0
        field = 11
        cuts = 2

        img = np.expand_dims(_get_image(300, 300), axis=-1)
        img = np.expand_dims(img, axis=0)

        redis_client = None
        storage = None
        consumer = consumers.ImageFileConsumer(redis_client, storage, 'predict')

        # image should be chopped into cuts**2 pieces and reassembled
        consumer.grpc_image = lambda x, y, z: x
        res = consumer.process_big_image(cuts, img, field, name, version)
        np.testing.assert_equal(res, img)

    def test__consume(self):
        prefix = 'prefix'
        status = 'new'
        redis_client = DummyRedis(prefix, status)
        storage = DummyStorage()
        consumer = consumers.ImageFileConsumer(redis_client, storage, 'predict')

        def _handle_error(err, rhash):  # pylint: disable=W0613
            raise err

        def grpc_image_multi(data, *args, **kwargs):  # pylint: disable=W0613
            return np.array(tuple(list(data.shape) + [2]))

        dummyhash = '{}:test.tiff:{}'.format(prefix, status)

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
        consumer = consumers.ImageFileConsumer(redis, storage, 'predict')
        consumer._handle_error = _handle_error
        consumer.grpc_image = grpc_image
        consumer._consume(dummyhash)


class TestZipFileConsumer(object):

    def test_is_valid_hash(self):
        items = ['item%s' % x for x in range(1, 4)]

        storage = DummyStorage()
        redis_client = DummyRedis(items)
        redis_client.hget = lambda *x: x[0]

        consumer = consumers.ZipFileConsumer(redis_client, storage, 'predict')
        assert consumer.is_valid_hash(None) is False
        assert consumer.is_valid_hash('file.ZIp') is True
        assert consumer.is_valid_hash('predict:1234567890:file.ZIp') is True
        assert consumer.is_valid_hash('track:123456789:file.zip') is True
        assert consumer.is_valid_hash('predict:123456789:file.zip') is True
        assert consumer.is_valid_hash('predict:1234567890:file.tiff') is False
        assert consumer.is_valid_hash('predict:1234567890:file.png') is False

    def test__upload_archived_images(self):
        N = 3
        items = ['item%s' % x for x in range(1, 4)]
        redis_client = DummyRedis(items)
        storage = DummyStorage(num=N)
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'predict')
        hsh = consumer._upload_archived_images(
            {'input_file_name': 'test.zip'}, 'predict:redis_hash:f.zip')
        assert len(hsh) == N

    def test__upload_finished_children(self):
        finished_children = ['predict:1.tiff', 'predict:2.zip', '']
        N = 3
        items = ['item%s' % x for x in range(1, N + 1)]
        redis_client = DummyRedis(items)
        storage = DummyStorage(num=N)
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'predict')
        path, url = consumer._upload_finished_children(
            finished_children, 'predict:redis_hash:f.zip')
        assert path and url

    def test__parse_failures(self):
        N = 3
        items = ['item%s' % x for x in range(1, N + 1)]
        redis_client = DummyRedis(items)
        storage = DummyStorage(num=N)
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'predict')

        # no failures
        failed_children = ''
        parsed = consumer._parse_failures(failed_children)
        assert parsed == ''

        failed_children = ['item1', 'item2', '']
        parsed = consumer._parse_failures(failed_children)
        assert 'item1=reason' in parsed and 'item2=reason' in parsed

    def test__consume(self):
        N = 3
        prefix = 'predict'
        items = ['item%s' % x for x in range(1, 4)]
        redis_client = DummyRedis(items)
        storage = DummyStorage(num=N)

        # test `status` = "new"
        status = 'new'
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'predict')
        consumer._upload_archived_images = lambda x, y: items
        dummyhash = '{queue}:{fname}.zip:{status}'.format(
            queue=prefix, status=status, fname=status)
        consumer._consume(dummyhash)

        # test `status` = "waiting"
        status = 'waiting'
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'predict')
        dummyhash = '{queue}:{fname}.zip:{status}'.format(
            queue=prefix, status=status, fname=status)
        consumer._consume(dummyhash)

        # test `status` = "cleanup"
        status = 'cleanup'
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'predict')
        consumer._upload_finished_children = lambda x, y: (x, y)
        dummyhash = '{queue}:{fname}.zip:{status}'.format(
            queue=prefix, status=status, fname=status)
        consumer._consume(dummyhash)

        # test `status` = "done"
        status = 'done'
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'predict')
        dummyhash = '{queue}:{fname}.zip:{status}'.format(
            queue=prefix, status=status, fname=status)
        consumer._consume(dummyhash)

        # test `status` = "failed"
        status = 'failed'
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'predict')
        dummyhash = '{queue}:{fname}.zip:{status}'.format(
            queue=prefix, status=status, fname=status)
        consumer._consume(dummyhash)


class TestTrackingConsumer(object):

    def test_is_valid_hash(self):
        items = ['item%s' % x for x in range(1, 4)]

        storage = DummyStorage()
        redis_client = DummyRedis(items)
        redis_client.hget = lambda *x: x[0]

        consumer = consumers.TrackingConsumer(redis_client, storage, 'track')
        assert consumer.is_valid_hash(None) is False
        assert consumer.is_valid_hash('predict:123456789:file.png') is False
        assert consumer.is_valid_hash('predict:1234567890:file.tiff') is True
        assert consumer.is_valid_hash('predict:1234567890:file.png') is False
        assert consumer.is_valid_hash('track:1234567890:file.ZIp') is False
        assert consumer.is_valid_hash('track:123456789:file.zip') is False
        assert consumer.is_valid_hash('track:1234567890:file.png') is False
        assert consumer.is_valid_hash('track:1234567890:file.tiff') is True
        assert consumer.is_valid_hash('track:1234567890:file.trk') is True
        assert consumer.is_valid_hash('track:1234567890:file.trks') is True
