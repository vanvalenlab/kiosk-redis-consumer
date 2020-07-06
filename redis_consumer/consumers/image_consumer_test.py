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

import itertools

import numpy as np

import pytest

from redis_consumer import consumers
from redis_consumer import settings
from redis_consumer.testing_utils import DummyStorage, redis_client, _get_image


class TestImageFileConsumer(object):
    # pylint: disable=R0201
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

    def test__get_processing_function(self):
        _funcs = settings.PROCESSING_FUNCTIONS
        settings.PROCESSING_FUNCTIONS = {
            'valid': {
                'valid': lambda x: True
            }
        }

        consumer = consumers.ImageFileConsumer(None, None, 'q')

        x = consumer._get_processing_function('VaLiD', 'vAlId')
        y = consumer._get_processing_function('vAlId', 'VaLiD')
        assert x == y

        with pytest.raises(ValueError):
            consumer._get_processing_function('invalid', 'valid')

        with pytest.raises(ValueError):
            consumer._get_processing_function('valid', 'invalid')

        settings.PROCESSING_FUNCTIONS = _funcs

    def test_process(self):
        _funcs = settings.PROCESSING_FUNCTIONS
        settings.PROCESSING_FUNCTIONS = {
            'valid': {
                'valid': lambda x: x
            }
        }

        img = np.zeros((1, 32, 32, 1))
        redis_client = DummyRedis([])
        consumer = consumers.ImageFileConsumer(redis_client, None, 'q')
        output = consumer.process(img, 'valid', 'valid')
        assert img.shape[1:] == output.shape

        settings.PROCESSING_FUNCTIONS = _funcs

    def test_detect_label(self):
        # pylint: disable=W0613
        redis_client = DummyRedis([])
        model_shape = (1, 216, 216, 1)
        consumer = consumers.ImageFileConsumer(redis_client, None, 'q')
        consumer.get_model_metadata = lambda x, y: {
            'in_tensor_dtype': 'DT_FLOAT',
            'in_tensor_shape': ','.join(str(s) for s in model_shape),
        }
        image = _get_image(model_shape[1] * 2, model_shape[2] * 2)

        settings.LABEL_DETECT_MODEL = 'dummymodel:1'

        def predict(*_, **__):
            data = np.zeros((3,))
            i = np.random.randint(3)
            data[i] = 1
            return data

        consumer.predict = predict

        settings.LABEL_DETECT_ENABLED = False

        label = consumer.detect_label(image)
        assert label is None

        settings.LABEL_DETECT_ENABLED = True

        label = consumer.detect_label(image)
        assert label in set(list(range(4)))

    def test_detect_scale(self):
        # pylint: disable=W0613
        redis_client = DummyRedis([])

        model_shape = (1, 216, 216, 1)
        consumer = consumers.ImageFileConsumer(redis_client, None, 'q')
        consumer.get_model_metadata = lambda x, y: {
            'in_tensor_dtype': 'DT_FLOAT',
            'in_tensor_shape': ','.join(str(s) for s in model_shape),
        }
        big_size = model_shape[1] * np.random.randint(2, 9)
        image = _get_image(big_size, big_size)

        expected = (model_shape[1] / (big_size)) ** 2

        settings.SCALE_DETECT_MODEL = 'dummymodel:1'

        def predict(*_, **__):
            sign = -1 if np.random.randint(1, 5) > 2 else 1
            return expected + sign * 1e-8  # small differences get averaged out

        consumer.predict = predict

        settings.SCALE_DETECT_ENABLED = False

        scale = consumer.detect_scale(image)
        assert scale == 1

        settings.SCALE_DETECT_ENABLED = True

        consumer.predict = predict

        scale = consumer.detect_scale(image)
        assert isinstance(scale, (float, int))
        np.testing.assert_almost_equal(scale, expected)

        # scale = consumer.detect_scale(np.expand_dims(image, axis=-1))
        # assert isinstance(scale, (float, int))
        # np.testing.assert_almost_equal(scale, expected)

    def test__consume(self):
        # pylint: disable=W0613
        prefix = 'predict'
        status = 'new'
        redis_client = DummyRedis(prefix, status)
        storage = DummyStorage()

        consumer = consumers.ImageFileConsumer(redis_client, storage, prefix)

        def _handle_error(err, rhash):
            raise err

        def grpc_image(data, *args, **kwargs):
            return data

        def grpc_image_multi(data, *args, **kwargs):
            return np.array(tuple(list(data.shape) + [2]))

        def grpc_image_list(data, *args, **kwargs):  # pylint: disable=W0613
            return [data, data]

        def detect_scale(_):
            return 1

        def detect_label(_):
            return 0

        def make_model_metadata_of_size(model_shape=(-1, 256, 256, 1)):

            def get_model_metadata(model_name, model_version):
                return [{
                    'in_tensor_name': 'image',
                    'in_tensor_dtype': 'DT_FLOAT',
                    'in_tensor_shape': ','.join(str(s) for s in model_shape),
                }]

            return get_model_metadata

        dummyhash = '{}:test.tiff:{}'.format(prefix, status)

        model_shapes = [
            (-1, 600, 600, 1),  # image too small, pad
            (-1, 300, 300, 1),  # image is exactly the right size
            (-1, 150, 150, 1),  # image too big, tile
        ]

        consumer._handle_error = _handle_error
        consumer.grpc_image = grpc_image
        consumer.detect_scale = detect_scale
        consumer.detect_label = detect_label

        # consumer.grpc_image = grpc_image_multi
        # consumer.get_model_metadata = make_model_metadata_of_size(model_shapes[0])
        #
        # result = consumer._consume(dummyhash)
        # assert result == consumer.final_status
        #
        # # test with a finished hash
        # result = consumer._consume('{}:test.tiff:{}'.format(prefix, 'done'))
        # assert result == 'done'

        for b in (False, True):
            settings.SCALE_DETECT_ENABLED = settings.LABEL_DETECT_ENABLED = b
            for model_shape in model_shapes:
                for grpc_func in (grpc_image, grpc_image_list):

                    consumer.grpc_image = grpc_func
                    consumer.get_model_metadata = \
                        make_model_metadata_of_size(model_shape)

                    result = consumer._consume(dummyhash)
                    assert result == consumer.final_status
                    # test with a finished hash
                    result = consumer._consume('{}:test.tiff:{}'.format(
                        prefix, consumer.final_status))
                    assert result == consumer.final_status

        # test with model_name and model_version
        redis_client.hgetall = lambda x: {
            'model_name': 'model',
            'model_version': '0',
            'label': '0',
            'scale': '1',
            'postprocess_function': '',
            'preprocess_function': '',
            'file_name': 'test_image.tiff',
            'input_file_name': 'test_image.tiff',
            'output_file_name': 'test_image.tiff'
        }
        redis_client.hmset = lambda x, y: True
        consumer = consumers.ImageFileConsumer(redis_client, storage, prefix)
        consumer._handle_error = _handle_error
        consumer.detect_scale = detect_scale
        consumer.detect_label = detect_label
        consumer.get_model_metadata = make_model_metadata_of_size((1, 300, 300, 1))
        consumer.grpc_image = grpc_image
        result = consumer._consume(dummyhash)
        assert result == consumer.final_status
