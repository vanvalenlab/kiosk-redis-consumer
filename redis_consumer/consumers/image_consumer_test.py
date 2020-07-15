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

import itertools

import numpy as np

import pytest

from redis_consumer import consumers
from redis_consumer import settings
from redis_consumer.testing_utils import DummyStorage, redis_client, _get_image


class TestImageFileConsumer(object):
    # pylint: disable=R0201,W0621
    def test_is_valid_hash(self, mocker, redis_client):
        storage = DummyStorage()
        mocker.patch.object(redis_client, 'hget', lambda x, y: x.split(':')[-1])

        consumer = consumers.ImageFileConsumer(redis_client, storage, 'predict')

        assert consumer.is_valid_hash(None) is False
        assert consumer.is_valid_hash('file.ZIp') is False
        assert consumer.is_valid_hash('predict:1234567890:file.ZIp') is False
        assert consumer.is_valid_hash('track:123456789:file.zip') is False
        assert consumer.is_valid_hash('predict:123456789:file.zip') is False
        assert consumer.is_valid_hash('predict:1234567890:file.tiff') is True
        assert consumer.is_valid_hash('predict:1234567890:file.png') is True

    def test_detect_label(self, mocker, redis_client):
        # pylint: disable=W0613
        model_shape = (1, 216, 216, 1)
        consumer = consumers.ImageFileConsumer(redis_client, None, 'q')

        def dummy_metadata(*_, **__):
            return {
                'in_tensor_dtype': 'DT_FLOAT',
                'in_tensor_shape': ','.join(str(s) for s in model_shape),
            }

        image = _get_image(model_shape[1] * 2, model_shape[2] * 2)

        def predict(*_, **__):
            data = np.zeros((3,))
            i = np.random.randint(3)
            data[i] = 1
            return data

        mocker.patch.object(consumer, 'predict', predict)
        mocker.patch.object(consumer, 'get_model_metadata', dummy_metadata)
        mocker.patch.object(settings, 'LABEL_DETECT_MODEL', 'dummymodel:1')

        mocker.patch.object(settings, 'LABEL_DETECT_ENABLED', False)
        label = consumer.detect_label(image)
        assert label is None

        mocker.patch.object(settings, 'LABEL_DETECT_ENABLED', True)
        label = consumer.detect_label(image)
        assert label in set(list(range(4)))

    def test_detect_scale(self, mocker, redis_client):
        # pylint: disable=W0613
        # TODO: test rescale is < 1% of the original
        model_shape = (1, 216, 216, 1)
        consumer = consumers.ImageFileConsumer(redis_client, None, 'q')

        def dummy_metadata(*_, **__):
            return {
                'in_tensor_dtype': 'DT_FLOAT',
                'in_tensor_shape': ','.join(str(s) for s in model_shape),
            }

        big_size = model_shape[1] * np.random.randint(2, 9)
        image = _get_image(big_size, big_size)

        expected = 1

        def predict(diff=1e-8):
            def _predict(*_, **__):
                sign = -1 if np.random.randint(1, 5) > 2 else 1
                return expected + sign * diff
            return _predict

        mocker.patch.object(consumer, 'get_model_metadata', dummy_metadata)
        mocker.patch.object(settings, 'SCALE_DETECT_ENABLED', False)
        mocker.patch.object(settings, 'SCALE_DETECT_MODEL', 'dummymodel:1')
        scale = consumer.detect_scale(image)
        assert scale == 1

        mocker.patch.object(settings, 'SCALE_DETECT_ENABLED', True)
        mocker.patch.object(consumer, 'predict', predict(1e-8))
        scale = consumer.detect_scale(image)
        # very small changes within error range:
        assert scale == 1

        mocker.patch.object(settings, 'SCALE_DETECT_ENABLED', True)
        mocker.patch.object(consumer, 'predict', predict(1e-1))
        scale = consumer.detect_scale(image)
        assert isinstance(scale, float)
        np.testing.assert_almost_equal(scale, expected, 1e-1)

    def test__consume(self, mocker, redis_client):
        # pylint: disable=W0613
        prefix = 'predict'
        status = 'new'
        storage = DummyStorage()

        consumer = consumers.ImageFileConsumer(redis_client, storage, prefix)

        def grpc_image(data, *args, **kwargs):
            return data

        def grpc_image_multi(data, *args, **kwargs):
            return np.array(tuple(list(data.shape) + [2]))

        def grpc_image_list(data, *args, **kwargs):  # pylint: disable=W0613
            return [data, data]

        def make_model_metadata_of_size(model_shape=(-1, 256, 256, 1)):

            def get_model_metadata(model_name, model_version):
                return [{
                    'in_tensor_name': 'image',
                    'in_tensor_dtype': 'DT_FLOAT',
                    'in_tensor_shape': ','.join(str(s) for s in model_shape),
                }]

            return get_model_metadata

        mocker.patch.object(consumer, 'detect_label', lambda x: 1)
        mocker.patch.object(consumer, 'detect_scale', lambda x: 1)
        mocker.patch.object(settings, 'LABEL_DETECT_ENABLED', True)

        grpc_funcs = (grpc_image, grpc_image_list)
        model_shapes = [
            (-1, 600, 600, 1),  # image too small, pad
            (-1, 300, 300, 1),  # image is exactly the right size
            (-1, 150, 150, 1),  # image too big, tile
        ]

        empty_data = {'input_file_name': 'file.tiff'}
        full_data = {
            'input_file_name': 'file.tiff',
            'model_version': '0',
            'model_name': 'model',
            'label': '1',
            'scale': '1',
        }
        label_no_model_data = full_data.copy()
        label_no_model_data['model_name'] = ''

        datasets = [empty_data, full_data, label_no_model_data]

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

        prod = itertools.product(model_shapes, grpc_funcs, datasets)

        for model_shape, grpc_func, data in prod:
            metadata = make_model_metadata_of_size(model_shape)
            mocker.patch.object(consumer, 'grpc_image', grpc_func)
            mocker.patch.object(consumer, 'get_model_metadata', metadata)
            mocker.patch.object(consumer, 'process', lambda *x: x[0])

            redis_client.hmset(test_hash, data)
            result = consumer._consume(test_hash)
            assert result == consumer.final_status
            result = redis_client.hget(test_hash, 'status')
            assert result == consumer.final_status
            test_hash += 1
