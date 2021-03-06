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
    def __init__(self, *_, **__):
        pass

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
    # pylint: disable=R0201,W0621
    def test_is_valid_hash(self, mocker, redis_client):
        queue = 'track'
        storage = DummyStorage()
        consumer = consumers.TrackingConsumer(redis_client, storage, queue)

        mocker.patch.object(redis_client, 'hget', lambda x, y: x.split(':')[-1])

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

    def test__update_progress(self, redis_client):
        queue = 'track'
        storage = DummyStorage()
        consumer = consumers.TrackingConsumer(redis_client, storage, queue)

        redis_hash = 'a job hash'
        progress = random.randint(0, 99)
        consumer._update_progress(redis_hash, progress)
        assert int(redis_client.hget(redis_hash, 'progress')) == progress

    def test__load_data(self, tmpdir, mocker, redis_client):
        queue = 'track'
        storage = DummyStorage()
        consumer = consumers.TrackingConsumer(redis_client, storage, queue)
        tmpdir = str(tmpdir)
        exp = random.randint(0, 99)

        # test load trk files
        key = 'trk file test'
        mocker.patch('redis_consumer.utils.load_track_file', lambda x: exp)
        result = consumer._load_data(key, tmpdir, 'data.trk')
        assert result == exp
        result = consumer._load_data(key, tmpdir, 'data.trks')
        assert result == exp

        # test bad filetype
        key = 'invalid filetype test'
        with pytest.raises(ValueError):
            consumer._load_data(key, tmpdir, 'data.npz')

        # test bad ndim for tiffstack
        fname = 'test.tiff'
        filepath = os.path.join(tmpdir, fname)
        tifffile.imsave(filepath, _get_image())
        with pytest.raises(ValueError):
            consumer._load_data(key, tmpdir, fname)

        # test successful workflow
        def hget_successful_status(*_):
            return consumer.final_status

        def hget_failed_status(*_):
            return consumer.failed_status

        def write_child_tiff(*_, **__):
            letters = string.ascii_lowercase
            name = ''.join(random.choice(letters) for i in range(12))
            path = os.path.join(tmpdir, '{}.tiff'.format(name))
            tifffile.imsave(path, _get_image(21, 21))
            return [path]

        mocker.patch.object(settings, 'INTERVAL', 0)
        mocker.patch.object(redis_client, 'hget', hget_successful_status)
        mocker.patch('redis_consumer.utils.iter_image_archive',
                     write_child_tiff)

        for label_detect in (True, False):
            mocker.patch.object(settings, 'SCALE_DETECT_ENABLED', label_detect)
            mocker.patch.object(settings, 'LABEL_DETECT_ENABLED', label_detect)

            tifffile.imsave(filepath, np.random.random((3, 21, 21)))
            results = consumer._load_data(key, tmpdir, fname)
            X, y = results.get('X'), results.get('y')
            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert X.shape == y.shape

        # test failed child
        with pytest.raises(RuntimeError):
            mocker.patch.object(redis_client, 'hget', hget_failed_status)
            consumer._load_data(key, tmpdir, fname)

        # test wrong number of images in the test file
        with pytest.raises(RuntimeError):
            mocker.patch.object(redis_client, 'hget', hget_successful_status)
            mocker.patch('redis_consumer.utils.iter_image_archive',
                         lambda *x: range(1, 3))
            consumer._load_data(key, tmpdir, fname)

    def test__get_tracker(self, mocker, redis_client):
        queue = 'track'
        storage = DummyStorage()

        shape = (5, 21, 21, 1)
        raw = np.random.random(shape)
        segmented = np.random.randint(1, 10, size=shape)

        mocker.patch.object(settings, 'NORMALIZE_TRACKING', True)
        consumer = consumers.TrackingConsumer(redis_client, storage, queue)
        tracker = consumer._get_tracker('item1', {}, raw, segmented)
        assert isinstance(tracker, redis_consumer.tracking.CellTracker)

    def test__consume(self, mocker, redis_client):
        queue = 'track'
        storage = DummyStorage()
        test_hash = 0

        consumer = consumers.TrackingConsumer(redis_client, storage, queue)

        mocker.patch.object(consumer, '_get_tracker', DummyTracker)
        mocker.patch.object(settings, 'DRIFT_CORRECT_ENABLED', True)

        frames = 3
        dummy_data = {
            'X': np.array([_get_image(21, 21) for _ in range(frames)]),
            'y': np.random.randint(0, 9, size=(frames, 21, 21)),
        }

        mocker.patch.object(consumer, '_load_data', lambda *x: dummy_data)

        # test finished statuses are returned
        for status in (consumer.failed_status, consumer.final_status):
            test_hash += 1
            data = {'input_file_name': 'file.tiff', 'status': status}
            redis_client.hmset(test_hash, data)
            result = consumer._consume(test_hash)
            assert result == status

        # test new key is processed
        test_hash += 1
        data = {'input_file_name': 'file.tiff', 'status': 'new'}
        redis_client.hmset(test_hash, data)
        result = consumer._consume(test_hash)
        assert result == consumer.final_status
        assert redis_client.hget(test_hash, 'status') == consumer.final_status
