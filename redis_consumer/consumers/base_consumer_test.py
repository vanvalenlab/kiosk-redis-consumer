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
"""Tests for base consumers"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import time

import numpy as np

import pytest

from redis_consumer import consumers
from redis_consumer import settings

from redis_consumer.testing_utils import Bunch, DummyStorage, redis_client


class TestConsumer(object):
    # pylint: disable=R0201
    def test_get_redis_hash(self):
        settings.EMPTY_QUEUE_TIMEOUT = 0.01  # don't sleep too long

        queue_name = 'q'
        # test emtpy queue
        items = []
        redis_client = DummyRedis(items, prefix=queue_name)
        consumer = consumers.Consumer(redis_client, None, queue_name)
        rhash = consumer.get_redis_hash()
        assert rhash is None

        # LLEN somehow gives stale data, should still be None
        redis_client.llen = lambda x: 1
        consumer = consumers.Consumer(redis_client, None, queue_name)
        rhash = consumer.get_redis_hash()
        assert rhash is None

        # test that invalid items are not processed and are removed.
        items = ['item%s:file.tif:new' % x for x in range(1, 4)]
        redis_client = DummyRedis(items, prefix=queue_name)

        consumer = consumers.Consumer(redis_client, None, queue_name)
        consumer.is_valid_hash = lambda x: x.startswith('item1')

        rhash = consumer.get_redis_hash()
        assert rhash == items[0]
        assert not redis_client.work_queue
        assert len(redis_client.processing_queue)
        assert redis_client.processing_queue == [rhash]

    def test_purge_processing_queue(self):
        queue_name = 'q'
        items = []
        redis_client = DummyRedis(items, prefix=queue_name)
        consumer = consumers.Consumer(redis_client, None, queue_name)

        redis_client.processing_queue = list(range(5))
        consumer.purge_processing_queue()
        assert not redis_client.processing_queue

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

    def test__put_back_hash(self):
        queue_name = 'q'

        # test emtpy queue
        redis_client = DummyRedis([], prefix=queue_name)
        consumer = consumers.Consumer(redis_client, None, queue_name)
        consumer._put_back_hash('DNE')  # should be None, shows warning

        # put back the proper item
        item = 'redis_hash1'
        redis_client.processing_queue = [item]
        consumer = consumers.Consumer(redis_client, None, queue_name)
        consumer._put_back_hash(item)

        # put back the wrong item
        redis_client.processing_queue = [item, 'otherhash']
        consumer = consumers.Consumer(redis_client, None, queue_name)
        consumer._put_back_hash(item)

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
            return 'done'

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

        consumer.consume()
        assert _processed is True

        _processed = False
        consumer.get_redis_hash = lambda: '{q}:f.tiff:{status}'.format(
            q=queue_name,
            status=consumer.final_status)
        consumer.consume()
        assert _processed is True

        _processed = 0

        def G(*_):
            global _processed
            _processed += 1
            return 'waiting'

        consumer._consume = G
        consumer.consume()
        assert _processed == N

    def test__consume(self):
        with np.testing.assert_raises(NotImplementedError):
            consumer = consumers.Consumer(None, None, 'q')
            consumer._consume('predict:new:hash.tiff')


class TestTensorFlowServingConsumer(object):
    # pylint: disable=R0201,W0613
    def test__get_predict_client(self):
        redis_client = DummyRedis([])
        consumer = consumers.TensorFlowServingConsumer(redis_client, None, 'q')

        with pytest.raises(ValueError):
            consumer._get_predict_client('model_name', 'model_version')

        consumer._get_predict_client('model_name', 1)

    def test_grpc_image(self):
        redis_client = DummyRedis([])
        consumer = consumers.TensorFlowServingConsumer(redis_client, None, 'q')
        model_shape = (-1, 128, 128, 1)

        def _get_predict_client(model_name, model_version):
            return Bunch(predict=lambda x, y: {
                'prediction': x[0]['data']
            })

        consumer._get_predict_client = _get_predict_client

        img = np.zeros((1, 32, 32, 3))
        out = consumer.grpc_image(img, 'model', 1, model_shape, 'i', 'DT_HALF')

        assert img.shape == out.shape
        assert img.sum() == out.sum()

        img = np.zeros((32, 32, 3))
        consumer._redis_hash = 'not none'
        out = consumer.grpc_image(img, 'model', 1, model_shape, 'i', 'DT_HALF')

        assert (1,) + img.shape == out.shape
        assert img.sum() == out.sum()

    def test_get_model_metadata(self):
        redis_client = DummyRedis([])
        model_shape = (-1, 216, 216, 1)
        model_dtype = 'DT_FLOAT'
        model_input_name = 'input_1'

        def hget_success(key, *others):
            metadata = {
                model_input_name: {
                    'dtype': model_dtype,
                    'tensorShape': {
                        'dim': [
                            {'size': str(x)}
                            for x in model_shape
                        ]
                    }
                }
            }
            return json.dumps(metadata)

        def hget_fail(key, *others):
            return None

        def _get_predict_client(model_name, model_version):
            return Bunch(get_model_metadata=lambda: {
                'metadata': {
                    'signature_def': {
                        'signatureDef': {
                            'serving_default': {
                                'inputs': {
                                    model_input_name: {
                                        'dtype': model_dtype,
                                        'tensorShape': {
                                            'dim': [
                                                {'size': str(x)}
                                                for x in model_shape
                                            ]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            })

        def _get_predict_client_multi(model_name, model_version):
            return Bunch(get_model_metadata=lambda: {
                'metadata': {
                    'signature_def': {
                        'signatureDef': {
                            'serving_default': {
                                'inputs': {
                                    model_input_name: {
                                        'dtype': model_dtype,
                                        'tensorShape': {
                                            'dim': [
                                                {'size': str(x)}
                                                for x in model_shape
                                            ]
                                        }
                                    },
                                    '{}_2'.format(model_input_name): {
                                        'dtype': model_dtype,
                                        'tensorShape': {
                                            'dim': [
                                                {'size': str(x)}
                                                for x in model_shape
                                            ]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            })

        def _get_bad_predict_client(model_name, model_version):
            return Bunch(get_model_metadata=lambda: dict())

        # test cached input
        redis_client.hget = hget_success
        consumer = consumers.TensorFlowServingConsumer(redis_client, None, 'q')
        consumer._get_predict_client = _get_predict_client
        metadata = consumer.get_model_metadata('model', 1)

        for m in metadata:
            assert m['in_tensor_dtype'] == model_dtype
            assert m['in_tensor_name'] == model_input_name
            assert m['in_tensor_shape'] == ','.join(str(x) for x in model_shape)

        # test stale cache
        redis_client.hget = hget_fail
        consumer = consumers.TensorFlowServingConsumer(redis_client, None, 'q')
        consumer._get_predict_client = _get_predict_client
        metadata = consumer.get_model_metadata('model', 1)

        for m in metadata:
            assert m['in_tensor_dtype'] == model_dtype
            assert m['in_tensor_name'] == model_input_name
            assert m['in_tensor_shape'] == ','.join(str(x) for x in model_shape)

        with pytest.raises(KeyError):
            redis_client.hget = hget_fail
            consumer = consumers.TensorFlowServingConsumer(redis_client, None, 'q')
            consumer._get_predict_client = _get_bad_predict_client
            consumer.get_model_metadata('model', 1)

    def test_predict(self):
        redis_client = DummyRedis([])
        consumer = consumers.TensorFlowServingConsumer(redis_client, None, 'q')

        def grpc_image(data, *args, **kwargs):
            return data

        def grpc_image_list(data, *args, **kwargs):  # pylint: disable=W0613
            return [data, data]

        model_shape = (-1, 128, 128, 1)

        image_shapes = [
            (256, 256, 1),
            (128, 128, 1),
            (64, 64, 1),
            (100, 100, 1),
            (300, 300, 1),
        ]

        for image_shape in image_shapes:
            for grpc_func in (grpc_image, grpc_image_list):
                for untile in (False, True):

                    x = np.random.random(image_shape)
                    consumer.grpc_image = grpc_func
                    consumer.get_model_metadata = lambda x, y: [{
                        'in_tensor_name': 'image',
                        'in_tensor_dtype': 'DT_HALF',
                        'in_tensor_shape': ','.join(str(s) for s in model_shape),
                    }]

                    consumer.predict(x, model_name='modelname', model_version=0,
                                     untile=untile)


class TestZipFileConsumer(object):
    # pylint: disable=R0201,W0613
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
            {'input_file_name': 'test.zip', 'children': ''},
            'predict:redis_hash:f.zip')
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

    def test__get_output_file_name(self):
        settings.GRPC_BACKOFF = 0
        redis_client = DummyRedis([])
        redis_client.ttl = lambda x: -1  # key is missing
        redis_client._update_masters_and_slaves = lambda: True

        redis_client._redis_master = Bunch(hget=lambda x, y: None)
        consumer = consumers.ZipFileConsumer(redis_client, None, 'predict')

        with pytest.raises(ValueError):
            redis_client.ttl = lambda x: -2  # key is missing
            consumer = consumers.ZipFileConsumer(redis_client, None, 'predict')
            consumer._get_output_file_name('randomkey')

        with pytest.raises(ValueError):
            redis_client.ttl = lambda x: 1  # key is expired
            consumer = consumers.ZipFileConsumer(redis_client, None, 'predict')
            consumer._get_output_file_name('randomkey')

        with pytest.raises(ValueError):
            redis_client.ttl = lambda x: -1  # key not expired
            consumer = consumers.ZipFileConsumer(redis_client, None, 'predict')
            consumer._get_output_file_name('randomkey')

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

    def test__cleanup(self):
        N = 3
        queue = 'predict'
        items = ['item%s' % x for x in range(1, N + 1)]
        redis_client = DummyRedis(items)
        storage = DummyStorage(num=N)
        consumer = consumers.ZipFileConsumer(redis_client, storage, queue)

        children = list('abcdef')
        done = ['{}:done'.format(c) for c in children[:3]]
        failed = ['{}:failed'.format(c) for c in children[3:]]

        consumer._cleanup(items[0], children, done, failed)

        # test non-float values
        redis_client = DummyRedis(items)
        redis_client.hmget = lambda *args: ['x' for a in args]
        consumer = consumers.ZipFileConsumer(redis_client, storage, queue)
        consumer._cleanup(items[0], children, done, failed)

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
        result = consumer._consume(dummyhash)
        assert result == 'waiting'

        # test `status` = "waiting"
        status = 'waiting'
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'predict')
        dummyhash = '{queue}:{fname}.zip:{status}'.format(
            queue=prefix, status=status, fname=status)
        result = consumer._consume(dummyhash)
        assert result == status

        # test `status` = "done"
        status = 'done'
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'predict')
        dummyhash = '{queue}:{fname}.zip:{status}'.format(
            queue=prefix, status=status, fname=status)
        result = consumer._consume(dummyhash)
        assert result == status

        # test `status` = "failed"
        status = 'failed'
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'predict')
        dummyhash = '{queue}:{fname}.zip:{status}'.format(
            queue=prefix, status=status, fname=status)
        result = consumer._consume(dummyhash)
        assert result == status

        # test `status` = "other-status"
        status = 'other-status'
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'predict')
        dummyhash = '{queue}:{fname}.zip:{status}'.format(
            queue=prefix, status=status, fname=status)
        result = consumer._consume(dummyhash)
        assert result == status
