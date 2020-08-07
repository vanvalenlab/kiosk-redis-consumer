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
"""Tests for Deepcell3DConsumer"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

import pytest

from redis_consumer import consumers
from redis_consumer.testing_utils import redis_client, DummyStorage


class TestDeepcell3DConsumer(object):
    # pylint: disable=R0201

    def test_is_valid_hash(self, mocker, redis_client):
        storage = DummyStorage()
        mocker.patch.object(redis_client, 'hget', lambda *x: x[0])

        consumer = consumers.Deepcell3DConsumer(redis_client, storage, 'deepcell3d')
        assert consumer.is_valid_hash(None) is False
        assert consumer.is_valid_hash('file.ZIp') is False
        assert consumer.is_valid_hash('predict:1234567890:file.ZIp') is False
        assert consumer.is_valid_hash('track:123456789:file.zip') is False
        assert consumer.is_valid_hash('predict:123456789:file.zip') is False
        assert consumer.is_valid_hash('multiplex:1234567890:file.tiff') is True
        assert consumer.is_valid_hash('multiplex:1234567890:file.png') is True

    def test__consume(self, mocker, redis_client):
        # pylint: disable=W0613

        def make_model_metadata_of_size(model_shape=(-1, 20, 256, 256, 1)):

            def get_model_metadata(model_name, model_version):
                return [{
                    'in_tensor_name': 'image',
                    'in_tensor_dtype': 'DT_FLOAT',
                    'in_tensor_shape': ','.join(str(s) for s in model_shape),
                }]
            return get_model_metadata

        def make_grpc_image(model_shape=(-1, 20, 256, 256, 1)):
            # pylint: disable=E1101
            shape = model_shape[1:-1]

            def grpc(data, *args, **kwargs):
                inner = np.random.random((1,) + shape + (1,))
                outer = np.random.random((1,) + shape + (1,))
                fgbg = np.random.random((1,) + shape + (2,))
                feature = np.random.random((1,) + shape + (3,))
                return [inner, outer, fgbg, feature]
            return grpc

        image_shape = (60, 300, 300, 1)
        model_shapes = [
            (-1, 40, 600, 600, 1),  # image too small, pad
            (-1, 20, 256, 256, 1),  # image is exactly the right size
            (-1, 10, 150, 150, 1),  # image too big, tile
        ]

        scales = ['.9', '']

        job_data = {
            'input_file_name': 'file.tiff',
        }

        consumer = consumers.Deepcell3DConsumer(redis_client, DummyStorage(), 'deepcell3d')

        test_hash = 0
        # test finished statuses are returned
        for status in (consumer.failed_status, consumer.final_status):
            test_hash += 1
            data = job_data.copy()
            data['status'] = status
            redis_client.hmset(test_hash, data)
            result = consumer._consume(test_hash)
            assert result == status
            result = redis_client.hget(test_hash, 'status')
            assert result == status
            test_hash += 1

        mocker.patch('redis_consumer.utils.get_image',
                     lambda x: np.random.random(image_shape))

        for model_shape, scale in itertools.product(model_shapes, scales):
            metadata = make_model_metadata_of_size(model_shape)
            grpc_image = make_grpc_image(model_shape)
            mocker.patch.object(consumer, 'get_model_metadata', metadata)
            mocker.patch.object(consumer, 'grpc_image', grpc_image)

            data = job_data.copy()
            data['scale'] = scale

            redis_client.hmset(test_hash, data)
            result = consumer._consume(test_hash)
            assert result == consumer.final_status
            result = redis_client.hget(test_hash, 'status')
            assert result == consumer.final_status
            test_hash += 1
