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
    # pylint: disable=R0201,W0621
    def test_get_redis_hash(self, mocker, redis_client):
        mocker.patch.object(settings, 'EMPTY_QUEUE_TIMEOUT', 0.01)
        queue_name = 'q'
        consumer = consumers.Consumer(redis_client, None, queue_name)

        # test emtpy queue
        assert consumer.get_redis_hash() is None

        # test that invalid items are not processed and are removed.
        item = 'item to process'
        redis_client.lpush(queue_name, item)
        # is_valid_hash returns True by default
        assert consumer.get_redis_hash() == item
        assert redis_client.llen(consumer.processing_queue) == 1
        assert redis_client.lpop(consumer.processing_queue) == item
        # queue should be empty, get None again
        assert consumer.get_redis_hash() is None

        # test invalid hash is failed and removed from queue
        mocker.patch.object(consumer, 'is_valid_hash', return_value=False)
        redis_client.lpush(queue_name, 'invalid')
        assert consumer.get_redis_hash() is None  # invalid hash, returns None
        # invalid has was removed from the processing queue
        assert redis_client.llen(consumer.processing_queue) == 0
        # invalid hash was not returend to the work queue
        assert redis_client.llen(consumer.queue) == 0

        # test llen returns 1 but item is gone by the time the pop happens
        mocker.patch.object(redis_client, 'llen', lambda x: 1)
        assert consumer.get_redis_hash() is None

    def test_purge_processing_queue(self, redis_client):
        queue_name = 'q'
        keys = ['abc', 'def', 'xyz']
        consumer = consumers.Consumer(redis_client, None, queue_name)
        # set keys in processing queue
        for key in keys:
            redis_client.lpush(consumer.processing_queue, key)

        consumer.purge_processing_queue()

        assert redis_client.llen(consumer.processing_queue) == 0
        assert redis_client.llen(consumer.queue) == len(keys)

    def test_update_key(self, redis_client):
        consumer = consumers.Consumer(redis_client, None, 'q')
        key = 'redis-hash'
        status = 'updated_status'
        new_value = 'some value'
        consumer.update_key(key, {
            'status': status,
            'new_field': new_value
        })
        redis_values = redis_client.hgetall(key)
        assert redis_values.get('status') == status
        assert redis_values.get('new_field') == new_value

        with pytest.raises(ValueError):
            consumer.update_key('redis-hash', 'data')

    def test_handle_error(self, redis_client):
        consumer = consumers.Consumer(redis_client, None, 'q')
        err = Exception('test exception')
        key = 'redis-hash'
        consumer._handle_error(err, key)

        redis_values = redis_client.hgetall(key)
        assert isinstance(redis_values, dict)
        assert 'status' in redis_values and 'reason' in redis_values
        assert redis_values.get('status') == 'failed'

    def test__put_back_hash(self, redis_client):
        queue_name = 'q'

        # test emtpy queue
        consumer = consumers.Consumer(redis_client, None, queue_name)
        consumer._put_back_hash('DNE')  # should be None, shows warning

        # put back the proper item
        item = 'redis_hash1'
        redis_client.lpush(consumer.processing_queue, item)
        consumer = consumers.Consumer(redis_client, None, queue_name)
        consumer._put_back_hash(item)
        assert redis_client.llen(consumer.processing_queue) == 0
        assert redis_client.llen(consumer.queue) == 1
        assert redis_client.lpop(consumer.queue) == item

        # put back the wrong item
        other = 'otherhash'
        redis_client.lpush(consumer.processing_queue, other, item)
        consumer = consumers.Consumer(redis_client, None, queue_name)
        consumer._put_back_hash(item)
        assert redis_client.llen(consumer.processing_queue) == 0
        assert redis_client.llen(consumer.queue) == 2
        assert redis_client.lpop(consumer.queue) == item
        assert redis_client.lpop(consumer.queue) == other

    def test_consume(self, mocker, redis_client):
        mocker.patch.object(settings, 'EMPTY_QUEUE_TIMEOUT', 0)
        queue_name = 'q'
        keys = [str(x) for x in range(1, 10)]
        err = OSError('thrown on purpose')
        i = 0

        consumer = consumers.Consumer(redis_client, DummyStorage(), queue_name)

        def throw_error(*_, **__):
            raise err

        def finish(*_, **__):
            return consumer.final_status

        def fail(*_, **__):
            return consumer.failed_status

        def in_progress(*_, **__):
            return 'another status'

        # empty redis queue
        spy = mocker.spy(time, 'sleep')
        assert redis_client.llen(consumer.queue) == 0
        consumer.consume()
        spy.assert_called_once_with(settings.EMPTY_QUEUE_TIMEOUT)

        # OK now let's fill the queue
        redis_client.lpush(consumer.queue, *keys)

        # test that _consume is called on each hash
        mocker.patch.object(consumer, '_consume', finish)
        spy = mocker.spy(consumer, '_consume')
        consumer.consume()
        spy.assert_called_once_with(keys[i])
        i += 1

        # error inside _consume calls _handle_error
        mocker.patch.object(consumer, '_consume', throw_error)
        spy = mocker.spy(consumer, '_handle_error')
        consumer.consume()
        spy.assert_called_once_with(err, keys[i])
        i += 1

        # status is in progress calls sleep
        mocker.patch.object(consumer, '_consume', in_progress)
        spy = mocker.spy(consumer, '_put_back_hash')
        consumer.consume()
        spy.assert_called_with(keys[i])
        i += 1

        # failed and done statuses call lrem
        spy = mocker.spy(redis_client, 'lrem')
        for status in (finish, fail):
            mocker.patch.object(consumer, '_consume', status)
            consumer.consume()
            spy.assert_called_with(consumer.processing_queue, 1, keys[i])
            i += 1

    def test__consume(self):
        with np.testing.assert_raises(NotImplementedError):
            consumer = consumers.Consumer(None, None, 'q')
            consumer._consume('predict:new:hash.tiff')


class TestTensorFlowServingConsumer(object):
    # pylint: disable=R0201,W0613,W0621
    def test__get_predict_client(self, redis_client):
        stg = DummyStorage()
        consumer = consumers.TensorFlowServingConsumer(redis_client, stg, 'q')

        with pytest.raises(ValueError):
            consumer._get_predict_client('model_name', 'model_version')

        consumer._get_predict_client('model_name', 1)

    def test_grpc_image(self, mocker, redis_client):
        storage = DummyStorage()
        queue = 'q'

        consumer = consumers.TensorFlowServingConsumer(
            redis_client, storage, queue)

        model_shape = (-1, 128, 128, 1)

        def _get_predict_client(model_name, model_version):
            return Bunch(predict=lambda x, y: {
                'prediction': x[0]['data']
            })

        mocker.patch.object(consumer, '_get_predict_client', _get_predict_client)

        img = np.zeros((1, 32, 32, 3))
        out = consumer.grpc_image(img, 'model', 1, model_shape, 'i', 'DT_HALF')
        assert img.shape == out.shape
        assert img.sum() == out.sum()

        img = np.zeros((32, 32, 3))
        consumer._redis_hash = 'not none'
        out = consumer.grpc_image(img, 'model', 1, model_shape, 'i', 'DT_HALF')
        assert (1,) + img.shape == out.shape
        assert img.sum() == out.sum()

    def test_get_model_metadata(self, mocker, redis_client):
        model_shape = (-1, 216, 216, 1)
        model_dtype = 'DT_FLOAT'
        model_input_name = 'input_1'
        model_version = 3
        model_name = 'good model'
        model = '{}:{}'.format(model_name, model_version)

        # load model metadata into client
        cached_metadata = {
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
        redis_client.hset(model, 'metadata', json.dumps(cached_metadata))

        def _get_predict_client(model_name, model_version):
            return Bunch(get_model_metadata=lambda: {
                'metadata': {
                    'signature_def': {
                        'signatureDef': {
                            'serving_default': {
                                'inputs': cached_metadata
                            }
                        }
                    }
                }
            })

        def _get_predict_client_multi(model_name, model_version):
            newdata = cached_metadata.copy()
            newdata['input_2'] = newdata[model_input_name]
            return Bunch(get_model_metadata=lambda: {
                'metadata': {
                    'signature_def': {
                        'signatureDef': {
                            'serving_default': {
                                'inputs': newdata
                            }
                        }
                    }
                }
            })

        def _get_bad_predict_client(model_name, model_version):
            return Bunch(get_model_metadata=dict)

        stg = DummyStorage()
        consumer = consumers.TensorFlowServingConsumer(redis_client, stg, 'q')

        for client in (_get_predict_client, _get_predict_client_multi):
            mocker.patch.object(consumer, '_get_predict_client', client)

            # test cached input
            metadata = consumer.get_model_metadata(model_name, model_version)
            for m in metadata:
                assert m['in_tensor_dtype'] == model_dtype
                assert m['in_tensor_name'] == model_input_name
                assert m['in_tensor_shape'] == ','.join(str(x) for x in
                                                        model_shape)

            # test stale cache
            metadata = consumer.get_model_metadata('another model', 0)
            for m in metadata:
                assert m['in_tensor_dtype'] == model_dtype
                assert m['in_tensor_name'] == model_input_name
                assert m['in_tensor_shape'] == ','.join(str(x) for x in
                                                        model_shape)

        with pytest.raises(KeyError):
            mocker.patch.object(consumer, '_get_predict_client',
                                _get_bad_predict_client)
            consumer.get_model_metadata('model', 1)

    def test_predict(self, mocker, redis_client):
        model_shape = (-1, 128, 128, 1)
        stg = DummyStorage()
        consumer = consumers.TensorFlowServingConsumer(redis_client, stg, 'q')

        mocker.patch.object(settings, 'TF_MAX_BATCH_SIZE', 2)
        mocker.patch.object(consumer, 'get_model_metadata', lambda x, y: [{
            'in_tensor_name': 'image',
            'in_tensor_dtype': 'DT_HALF',
            'in_tensor_shape': ','.join(str(s) for s in model_shape),
        }])

        def grpc_image(data, *args, **kwargs):
            return data

        def grpc_image_list(data, *args, **kwargs):  # pylint: disable=W0613
            return [data, data]

        image_shapes = [
            (256, 256, 1),
            (128, 128, 1),
            (64, 64, 1),
            (100, 100, 1),
            (300, 300, 1),
            (257, 301, 1),
            (65, 127, 1),
        ]
        grpc_funcs = (grpc_image, grpc_image_list)
        untiles = (False, True)
        prod = itertools.product(image_shapes, grpc_funcs, untiles)

        for image_shape, grpc_func, untile in prod:
            x = np.random.random(image_shape)
            mocker.patch.object(consumer, 'grpc_image', grpc_func)

            consumer.predict(x, model_name='modelname', model_version=0,
                             untile=untile)

        # test mismatch of input data and model shape
        with pytest.raises(ValueError):
            x = np.random.random((5,))
            consumer.predict(x, model_name='modelname', model_version=0)

        # test multiple model metadata inputs are not supported
        with pytest.raises(ValueError):
            mocker.patch.object(consumer, 'get_model_metadata', grpc_image_list)
            x = np.random.random((300, 300, 1))
            consumer.predict(x, model_name='modelname', model_version=0)


class TestZipFileConsumer(object):
    # pylint: disable=R0201,W0613,W0621
    def test_is_valid_hash(self, mocker, redis_client):
        consumer = consumers.ZipFileConsumer(
            redis_client, DummyStorage(), 'predict')

        def get_file_from_hash(redis_hash, _):
            return redis_hash.split(':')[-1]

        mocker.patch.object(redis_client, 'hget', get_file_from_hash)

        assert consumer.is_valid_hash(None) is False
        assert consumer.is_valid_hash('file.ZIp') is True
        assert consumer.is_valid_hash('predict:1234567890:file.ZIp') is True
        assert consumer.is_valid_hash('track:123456789:file.zip') is True
        assert consumer.is_valid_hash('predict:123456789:file.zip') is True
        assert consumer.is_valid_hash('predict:1234567890:file.tiff') is False
        assert consumer.is_valid_hash('predict:1234567890:file.png') is False

    def test__upload_archived_images(self, mocker, redis_client):
        N = 3
        storage = DummyStorage(num=N)
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'predict')
        # mocker.patch.object(consumer.storage, 'download')
        hvalues = {'input_file_name': 'test.zip', 'children': 'none'}
        redis_hash = 'predict:redis_hash:f.zip'
        hsh = consumer._upload_archived_images(hvalues, redis_hash)
        assert len(hsh) == N

    def test__upload_finished_children(self, mocker, redis_client):
        finished_children = ['predict:1.tiff', 'predict:2.zip', '']
        N = 3
        storage = DummyStorage(num=N)
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'predict')
        mocker.patch.object(consumer, '_get_output_file_name', lambda x: x)

        path, url = consumer._upload_finished_children(
            finished_children, 'predict:redis_hash:f.zip')
        assert path and url

    def test__get_output_file_name(self, mocker, redis_client):
        # TODO: bad coverage
        mocker.patch.object(settings, 'GRPC_BACKOFF', 0)
        storage = DummyStorage()
        queue = 'q'

        consumer = consumers.ZipFileConsumer(redis_client, storage, queue)

        # happy path
        key = 'some key'
        expected = 'output.zip'
        redis_client.hset(key, 'output_file_name', expected)
        assert consumer._get_output_file_name(key) == expected

        # handling missing output file
        key = 'key without output file'
        spy = mocker.spy(redis_client, 'ttl')

        # add key without output file but before it is expired
        redis_client.hset(key, 'some field', 'some value')
        with pytest.raises(ValueError):
            consumer._get_output_file_name(key)
        assert spy.spy_return == -1

        # expire key
        redis_client.expire(key, 10)  # TTL should be -1
        with pytest.raises(ValueError):
            consumer._get_output_file_name(key)
        assert spy.spy_return == 10

        # key does not exist
        with pytest.raises(ValueError):
            consumer._get_output_file_name('randomkey')  # TTL should be -2
        assert spy.spy_return == -2

    def test__parse_failures(self, mocker, redis_client):
        N = 3
        storage = DummyStorage(num=N)

        keys = [str(x) for x in range(4)]
        consumer = consumers.ZipFileConsumer(redis_client, storage, 'predict')
        for key in keys:
            redis_client.lpush(consumer.queue, key)
            redis_client.hset(key, 'reason', 'reason{}'.format(key))

        spy = mocker.spy(redis_client, 'hget')
        parsed = consumer._parse_failures(keys)
        spy.assert_called_with(keys[-1], 'reason')
        for key in keys:
            assert '{0}=reason{0}'.format(key) in parsed

        # no failures
        failed_children = ['']
        parsed = consumer._parse_failures(failed_children)
        assert parsed == ''

    def test__cleanup(self, mocker, redis_client):
        N = 3
        queue = 'predict'
        done = [str(i) for i in range(N)]
        failed = [str(i) for i in range(N + 1, N * 2)]
        storage = DummyStorage(num=N)
        consumer = consumers.ZipFileConsumer(redis_client, storage, queue)

        redis_hash = 'some job hash'

        mocker.patch.object(consumer, '_upload_finished_children',
                            lambda *x: ('a', 'b'))

        for item in done:
            redis_client.hset(item, 'total_time', 1)  # summary field
        for item in failed:
            redis_client.hset(item, 'reason', 1)  # summary field

        children = done + failed

        consumer._cleanup(redis_hash, children, done, failed)

        assert redis_client.hget(redis_hash, 'total_jobs') == str(len(children))
        for key in children:
            assert redis_client.ttl(key) > 0  # all keys are expired

    def test__consume(self, mocker, redis_client):
        N = 3
        storage = DummyStorage(num=N)
        children = list('abcdefg')
        queue = 'q'
        test_hash = 0
        consumer = consumers.ZipFileConsumer(redis_client, storage, queue)

        # test finished statuses are returned
        for status in (consumer.failed_status, consumer.final_status, 'weird'):
            test_hash += 1
            redis_client.hset(test_hash, 'status', status)
            result = consumer._consume(test_hash)
            assert result == status

        # test `status` = "new"
        dummy = lambda *x: children
        mocker.patch.object(consumer, '_cleanup', dummy)
        mocker.patch.object(consumer, '_upload_archived_images', dummy)
        mocker.patch.object(consumer, '_upload_finished_children', dummy)

        test_hash += 1
        redis_client.hset(test_hash, 'status', 'new')
        result = consumer._consume(test_hash)
        assert result == 'waiting'
        assert redis_client.hget(test_hash, 'children') == ','.join(children)

        # test `status` = "waiting"
        status = 'waiting'
        test_hash += 1
        children = ['done', 'failed', 'waiting', 'move-to-done']
        child_statuses = ['done', 'failed', 'waiting', 'done']
        data = {'status': status, 'children': ','.join(children)}
        redis_client.hmset(test_hash, data)
        for child, child_status in zip(children, child_statuses):
            redis_client.hset(child, 'status', child_status)

        result = consumer._consume(test_hash)
        assert result == status
        hvals = redis_client.hgetall(test_hash)
        done_children = set(hvals.get('children:done', '').split(','))
        assert done_children == {'done', 'move-to-done'}
        assert hvals.get('children:failed') == 'failed'

        # set the "waiting" child to "done"
        redis_client.hset('waiting', 'status', consumer.final_status)
        result = consumer._consume(test_hash)
        assert result == consumer.final_status
        hvals = redis_client.hgetall(test_hash)
        done_children = set(hvals.get('children:done', '').split(','))
        assert done_children == {'done', 'move-to-done', 'waiting'}
        assert hvals.get('children:failed') == 'failed'
