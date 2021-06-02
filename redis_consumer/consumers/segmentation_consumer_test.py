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
"""Tests for post-processing functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np

import pytest

from redis_consumer import consumers
from redis_consumer import settings

from redis_consumer.testing_utils import Bunch
from redis_consumer.testing_utils import DummyStorage
from redis_consumer.testing_utils import redis_client
from redis_consumer.testing_utils import _get_image


class TestSegmentationConsumer(object):
    # pylint: disable=R0201,W0621

    def test_detect_label(self, mocker, redis_client):
        # pylint: disable=W0613
        shape = (1, 256, 256, 1)
        queue = 'q'
        consumer = consumers.SegmentationConsumer(redis_client, None, queue)

        expected_label = random.randint(1, 9)

        mock_app = Bunch(
            predict=lambda *x, **y: expected_label,
            model=Bunch(get_batch_size=lambda *x: 1))

        mocker.patch.object(consumer, 'get_grpc_app', lambda *x: mock_app)

        image = _get_image(shape[1] * 2, shape[2] * 2, shape[3])

        mocker.patch.object(settings, 'LABEL_DETECT_ENABLED', False)
        label = consumer.detect_label(image)
        assert label == 0

        mocker.patch.object(settings, 'LABEL_DETECT_ENABLED', True)
        label = consumer.detect_label(image)
        assert label == expected_label

    def test_get_image_label(self, mocker, redis_client):
        queue = 'q'
        stg = DummyStorage()
        consumer = consumers.SegmentationConsumer(redis_client, stg, queue)
        image = _get_image(256, 256, 1)

        # test no label provided
        expected = 1
        mocker.patch.object(consumer, 'detect_label', lambda *x: expected)
        label = consumer.get_image_label(None, image, 'some hash')
        assert label == expected

        # test label provided
        expected = 2
        label = consumer.get_image_label(expected, image, 'some hash')
        assert label == expected

        # test label provided is invalid
        with pytest.raises(ValueError):
            label = -1
            consumer.get_image_label(label, image, 'some hash')

        # test label provided is bad type
        with pytest.raises(ValueError):
            label = 'badval'
            consumer.get_image_label(label, image, 'some hash')

    def test__consume_finished_status(self, redis_client):
        queue = 'q'
        storage = DummyStorage()

        consumer = consumers.SegmentationConsumer(redis_client, storage, queue)

        empty_data = {'input_file_name': 'file.tiff'}

        test_hash = 0
        # test finished statuses are returned
        for status in (consumer.failed_status, consumer.final_status):
            test_hash += 1
            data = empty_data.copy()
            data['status'] = status
            redis_client.hmset(test_hash, data)
            result = consumer._consume(test_hash)
            assert result == status
            result = redis_client.hget(test_hash, 'status')
            assert result == status
            test_hash += 1

    def test__consume(self, mocker, redis_client):
        # pylint: disable=W0613
        queue = 'predict'
        storage = DummyStorage()

        consumer = consumers.SegmentationConsumer(redis_client, storage, queue)

        empty_data = {'input_file_name': 'file.tiff'}

        output_shape = (1, 32, 32, 1)

        mock_app = Bunch(
            predict=lambda *x, **y: np.random.randint(1, 5, size=output_shape),
            model_mpp=1,
            model=Bunch(
                get_batch_size=lambda *x: 1,
                input_shape=(1, 32, 32, 1)
            )
        )

        mocker.patch.object(consumer, 'get_grpc_app', lambda *x, **_: mock_app)
        mocker.patch.object(consumer, 'get_image_scale', lambda *x, **_: 1)
        mocker.patch.object(consumer, 'get_image_label', lambda *x, **_: 1)
        mocker.patch.object(consumer, 'validate_model_input', lambda *x, **_: True)
        mocker.patch.object(consumer, 'detect_dimension_order', lambda *x, **_: 'YXC')

        test_hash = 'some hash'

        redis_client.hmset(test_hash, empty_data)
        result = consumer._consume(test_hash)
        assert result == consumer.final_status
        result = redis_client.hget(test_hash, 'status')
        assert result == consumer.final_status
