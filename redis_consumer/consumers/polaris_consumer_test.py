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
from cProfile import label

import numpy as np

import os
import pytest
import tempfile
import tifffile
import uuid

from redis_consumer import consumers
from redis_consumer import settings
from redis_consumer import utils
from redis_consumer.testing_utils import _get_image
from redis_consumer.testing_utils import Bunch
from redis_consumer.testing_utils import DummyStorage
from redis_consumer.testing_utils import redis_client


class TestPolarisConsumer(object):
    # pylint: disable=R0201,W0621

    def test__add_images(self, redis_client):
        queue = 'polaris'
        storage = DummyStorage()
        consumer = consumers.PolarisConsumer(redis_client, storage, queue)

        test_im = np.random.random(size=(1, 32, 32, 1))
        test_im_name = 'test_im'

        test_hvals = {'original_name': test_im_name}
        uid = uuid.uuid4().hex

        test_im_hash = consumer._add_images(test_hvals, uid, test_im, queue)
        split_hash = test_im_hash.split(":")

        assert split_hash[0] == queue
        assert split_hash[1] == '{}-{}-{}-image.tif'.format(uid,
                                                            test_hvals.get('original_name'),
                                                            queue)

        result = redis_client.hget(test_im_hash, 'status')
        assert result == 'new'

    def test__analyze_images(self, redis_client):
        queue = 'polaris'
        storage = DummyStorage()
        consumer = consumers.PolarisConsumer(redis_client, storage, queue)

        with tempfile.TemporaryDirectory() as tempdir:
            test_hash = 'test hash'
            fname = 'file.tiff'
            input_size = (1, 32, 32, 1)
            tifffile.imsave(os.path.join(tempdir, fname),
                            np.random.random(size=input_size))

            empty_data = {'input_file_name': 'file.tiff',
                          'segmentation_type': 'cell culture',
                          'channels': '0,1,2'}
            redis_client.hmset(test_hash, empty_data)
            res = consumer._analyze_images(test_hash, tempdir, fname)
        assert np.shape(res['segmentation']) == input_size

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

    def test__consume(self, mocker, redis_client):
        # pylint: disable=W0613
        queue = 'polaris'
        storage = DummyStorage()

        consumer = consumers.PolarisConsumer(redis_client, storage, queue)

        # consume with segmentation and spot detection
        empty_data = {'input_file_name': 'file.tiff',
                      'segmentation_type': 'cell culture'}
        mocker.patch.object(consumer,
                            '_analyze_images',
                            lambda *x, **_: {'coords': np.random.randint(32, size=(1, 10, 2)),
                                             'segmentation': np.random.random(size=(1, 32, 32, 1))
                                             }
                            )
        test_hash = 'some hash'
        redis_client.hmset(test_hash, empty_data)
        result = consumer._consume(test_hash)
        assert result == consumer.final_status
        result = redis_client.hget(test_hash, 'status')
        assert result == consumer.final_status

        # consume with spot detection only
        empty_data = {'input_file_name': 'file.tiff',
                      'segmentation_type': 'none'}
        mocker.patch.object(consumer,
                            '_analyze_images',
                            lambda *x, **_: {'coords': np.random.randint(32, size=(1, 10, 2)),
                                             'segmentation': []})
        test_hash = 'some other hash'
        redis_client.hmset(test_hash, empty_data)
        result = consumer._consume(test_hash)
        assert result == consumer.final_status
        result = redis_client.hget(test_hash, 'status')
        assert result == consumer.final_status
