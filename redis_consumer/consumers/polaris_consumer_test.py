# Copyright 2016-2022 The Van Valen Lab at the California Institute of
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
"""Tests for PolarisConsumer"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import pytest

from redis_consumer import consumers
from redis_consumer import settings
from redis_consumer.testing_utils import _get_image
from redis_consumer.testing_utils import Bunch
from redis_consumer.testing_utils import DummyStorage
from redis_consumer.testing_utils import redis_client


class TestPolarisConsumer(object):
    # pylint: disable=R0201,W0621

    def test__consume_finished_status(self, redis_client):
        queue = 'q'
        storage = DummyStorage()

        consumer = consumers.PolarisConsumer(redis_client, storage, queue)

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

    # def test__consume(self, mocker, redis_client):
    #     # pylint: disable=W0613
    #     queue = 'polaris'
    #     storage = DummyStorage()

    #     consumer = consumers.PolarisConsumer(redis_client, storage, queue)

    #     empty_data = {'input_file_name': 'file.tiff'}

    #     output_shape = (1, 256, 256, 2)

    #     mock_app = Bunch(
    #         predict=lambda *x, **y: np.random.randint(1, 5, size=output_shape),
    #         model_mpp=1,
    #         model=Bunch(
    #             get_batch_size=lambda *x: 1,
    #             input_shape=(1, 32, 32, 1)
    #         )
    #     )

    #     mocker.patch.object(consumer, 'get_grpc_app', lambda *x, **_: mock_app)
    #     mocker.patch.object(consumer, 'get_image_scale', lambda *x, **_: 1)
    #     mocker.patch.object(consumer, 'validate_model_input', lambda *x, **_: x[0])
    #     mocker.patch.object(consumer, 'detect_dimension_order', lambda *x, **_: 'YXC')

    #     test_hash = 'some hash'

    #     redis_client.hmset(test_hash, empty_data)
    #     result = consumer._consume(test_hash)
    #     assert result == consumer.final_status
    #     result = redis_client.hget(test_hash, 'status')
    #     assert result == consumer.final_status
