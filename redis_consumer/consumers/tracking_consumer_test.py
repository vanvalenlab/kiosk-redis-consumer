# Copyright 2016-2020 The Van Valen Lab at the California Institute of
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

import numpy as np
from skimage.external import tifffile as tiff

import pytest

from redis_consumer import consumers
from redis_consumer import settings
from redis_consumer import utils


def _get_image(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h) * 64
    variance = np.random.rand(img_w, img_h) * (255 - 64)
    img = np.random.rand(img_w, img_h) * variance + bias
    return img


class Bunch(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


class DummyRedis(object):
    # pylint: disable=R0201,W0613
    def __init__(self, items=None, prefix='predict', status='new'):
        items = [] if items is None else items
        self.work_queue = copy.copy(items)
        self.processing_queue = []
        self.prefix = '/'.join(x for x in prefix.split('/') if x)
        self.status = status
        self._redis_master = self
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
    # pylint: disable=R0201,W0613
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

    def upload(self, zip_path, subdir=None):
        return True, True

    def get_public_url(self, zip_path):
        return True


class DummyTracker(object):
    # pylint: disable=R0201,W0613
    def _track_cells(self):
        return None

    def track_cells(self):
        return None

    def dump(self, *_, **__):
        return None

    def postprocess(self, *_, **__):
        return {
            'y_tracked': np.zeros((32, 32, 1)),
            'tracks': []
        }


class TestTrackingConsumer(object):
    # pylint: disable=R0201
    def test_is_valid_hash(self):
        queue = 'track'
        items = ['item%s' % x for x in range(1, 4)]

        storage = DummyStorage()
        redis_client = DummyRedis(items)
        redis_client.hget = lambda *x: x[0]

        consumer = consumers.TrackingConsumer(redis_client, storage, queue)
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

    def test__get_tracker(self):
        queue = 'track'
        items = ['item%s' % x for x in range(1, 4)]

        storage = DummyStorage()
        redis_client = DummyRedis(items)
        redis_client.hget = lambda *x: x[0]

        shape = (5, 21, 21, 1)
        raw = np.random.random(shape)
        segmented = np.random.randint(1, 10, size=shape)

        settings.NORMALIZE_TRACKING = True

        consumer = consumers.TrackingConsumer(redis_client, storage, queue)
        consumer._get_tracker('item1', {}, raw, segmented)

    def test__consume(self):
        queue = 'track'
        items = ['item%s' % x for x in range(1, 4)]

        storage = DummyStorage()
        redis_client = DummyRedis(items)
        redis_client.hget = lambda *x: x[0]

        # test short-circuit _consume()
        consumer = consumers.TrackingConsumer(redis_client, storage, queue)

        status = 'done'
        dummyhash = '{queue}:{fname}.zip:{status}'.format(
            queue=queue, status=status, fname=status)

        result = consumer._consume(dummyhash)
        assert result == status

        # test valid _consume flow
        status = 'new'
        dummyhash = '{queue}:{fname}.zip:{status}'.format(
            queue=queue, status=status, fname=status)
        dummy_data = np.zeros((1, 1, 1))
        consumer._load_data = lambda *x: {'X': dummy_data, 'y': dummy_data}
        consumer._get_tracker = lambda *args: DummyTracker()
        result = consumer._consume(dummyhash)
        assert result == consumer.final_status
