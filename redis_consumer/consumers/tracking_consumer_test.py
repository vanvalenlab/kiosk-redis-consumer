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
import random
import string

import pytest

import numpy as np
from skimage.external import tifffile

import redis_consumer
from redis_consumer import consumers
from redis_consumer import settings
from redis_consumer.testing_utils import DummyStorage, redis_client, _get_image


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

    def test__load_data(self, tmpdir):
        queue = 'track'
        items = ['item%s' % x for x in range(1, 4)]
        redis_hash = 'track:1234567890:file.trks'
        storage = DummyStorage()
        redis_client = DummyRedis(items)
        consumer = consumers.TrackingConsumer(redis_client, storage, queue)

        # test bad filetype
        with pytest.raises(ValueError):
            consumer._load_data(redis_hash, str(tmpdir), 'data.npz')

        # TODO: test successful workflow

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
