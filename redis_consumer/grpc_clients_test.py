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
"""Tests for gRPC Clients"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging

import pytest

import numpy as np
from google.protobuf.json_format import MessageToJson
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow_serving.apis.predict_pb2 import PredictResponse
from tensorflow_serving.apis.get_model_metadata_pb2 import GetModelMetadataResponse

from redis_consumer.testing_utils import _get_image, make_model_metadata_of_size

from redis_consumer import grpc_clients, settings


class DummyPredictClient(object):
    # pylint: disable=unused-argument
    def __init__(self, host, model_name, model_version):
        self.logger = logging.getLogger(self.__class__.__name__)

    def predict(self, request_data, request_timeout=10):
        retval = {}
        for i, d in enumerate(request_data):
            retval['prediction{}'.format(i)] = d.get('data')
        return retval


def test_make_tensor_proto():
    # test with numpy array
    data = _get_image(300, 300, 1)
    proto = grpc_clients.make_tensor_proto(data, 'DT_FLOAT')
    assert isinstance(proto, (TensorProto,))
    # test with value
    data = 10.0
    proto = grpc_clients.make_tensor_proto(data, types_pb2.DT_FLOAT)
    assert isinstance(proto, (TensorProto,))


def test_grpc_response_to_dict():
    # pylint: disable=E1101
    # test valid response
    data = _get_image(300, 300, 1)
    tensor_proto = grpc_clients.make_tensor_proto(data, 'DT_FLOAT')
    response = PredictResponse()
    response.outputs['prediction'].CopyFrom(tensor_proto)
    response_dict = grpc_clients.grpc_response_to_dict(response)
    assert isinstance(response_dict, (dict,))
    np.testing.assert_allclose(response_dict['prediction'], data)
    # test scalar input
    data = 3
    tensor_proto = grpc_clients.make_tensor_proto(data, 'DT_FLOAT')
    response = PredictResponse()
    response.outputs['prediction'].CopyFrom(tensor_proto)
    response_dict = grpc_clients.grpc_response_to_dict(response)
    assert isinstance(response_dict, (dict,))
    np.testing.assert_allclose(response_dict['prediction'], data)
    # test bad dtype
    # logs an error, but should throw a KeyError as well.
    data = _get_image(300, 300, 1)
    tensor_proto = grpc_clients.make_tensor_proto(data, 'DT_FLOAT')
    response = PredictResponse()
    response.outputs['prediction'].CopyFrom(tensor_proto)
    response.outputs['prediction'].dtype = 32

    with pytest.raises(KeyError):
        response_dict = grpc_clients.grpc_response_to_dict(response)


class TestPredictClient(object):

    def test_predict(self, mocker):
        name = 'test model name'
        version = 3
        client = grpc_clients.PredictClient('host', name, version)

        arr = np.random.random((4, 4, 1))
        tensor_name = 'test_input'
        tensor_dtype = 'DT_FLOAT'

        # just return the request data
        mock_retry_grpc = lambda request, _: request
        mocker.patch.object(client, '_retry_grpc', mock_retry_grpc)

        mocker.patch('redis_consumer.grpc_clients.grpc_response_to_dict',
                     lambda x: json.loads(MessageToJson(x)))

        req_data = [{
            'data': arr,
            'in_tensor_name': tensor_name,
            'in_tensor_dtype': tensor_dtype,
        }]

        request_json = client.predict(req_data)

        # response should be json formatted
        assert isinstance(request_json, dict)
        # confirm that the request is properly formatted
        assert request_json['modelSpec']['name'] == name
        assert request_json['modelSpec']['version'] == str(version)
        assert tensor_name in request_json['inputs']
        assert request_json['inputs'][tensor_name]['dtype'] == tensor_dtype

    def test_get_model_metadata(self, mocker):
        name = 'test model name'
        version = 3
        client = grpc_clients.PredictClient('host', name, version)

        # just return the request data
        mock_retry_grpc = lambda request, _: request
        mocker.patch.object(client, '_retry_grpc', mock_retry_grpc)

        metadata = client.get_model_metadata()
        # response should be json formatted
        assert isinstance(metadata, dict)
        # confirm that the request is properly formatted
        assert metadata['modelSpec']['name'] == name
        assert metadata['modelSpec']['version'] == str(version)


class TestGrpcModelWrapper(object):
    shape = (1, 300, 300, 1)
    name = 'test-model'
    version = '0'

    def _get_metadata(self):
        metadata_fn = make_model_metadata_of_size(self.shape)
        return metadata_fn(self.name, self.version)

    def test_init(self):
        metadata = self._get_metadata()
        wrapper = grpc_clients.GrpcModelWrapper(None, metadata)
        assert wrapper.input_shape == self.shape

        metadata += metadata
        wrapper = grpc_clients.GrpcModelWrapper(None, metadata)
        assert isinstance(wrapper.input_shape, list)
        for s in wrapper.input_shape:
            assert s == self.shape

    def test_get_batch_size(self, mocker):
        metadata = self._get_metadata()
        wrapper = grpc_clients.GrpcModelWrapper(None, metadata)

        for m in (.5, 1, 2):
            mocker.patch.object(settings, 'TF_MIN_MODEL_SIZE', self.shape[1] * m)
            batch_size = wrapper.get_batch_size()
            assert batch_size == settings.TF_MAX_BATCH_SIZE * m * m

        # what if the model has no defined size except the channel
        channels = 3
        shape = [(-1, -1, -1, -1, channels)]
        mocker.patch.object(wrapper, 'input_shape', shape)
        batch_size = wrapper.get_batch_size()
        assert batch_size == settings.TF_MAX_BATCH_SIZE // channels

    def test_send_grpc(self):
        client = DummyPredictClient(1, 2, 3)
        metadata = self._get_metadata()
        metadata[0]['in_tensor_dtype'] = 'DT_HALF'
        wrapper = grpc_clients.GrpcModelWrapper(client, metadata)

        input_data = np.ones(self.shape)
        result = wrapper.send_grpc(input_data)
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], input_data)

        # test multiple inputs
        metadata = self._get_metadata() + self._get_metadata()
        input_data = [np.ones(self.shape)] * 2
        wrapper = grpc_clients.GrpcModelWrapper(client, metadata)
        result = wrapper.send_grpc(input_data)
        assert isinstance(result, list)
        assert len(result) == 2
        np.testing.assert_array_equal(result, input_data)

        # test inputs don't match metadata
        with pytest.raises(ValueError):
            wrapper.send_grpc(np.ones(self.shape))

    def test_predict(self, mocker):
        metadata = self._get_metadata()
        wrapper = grpc_clients.GrpcModelWrapper(None, metadata)

        def mock_send_grpc(img):
            return img if isinstance(img, list) else [img]

        mocker.patch.object(wrapper, 'send_grpc', mock_send_grpc)

        batch_size = 2
        input_data = np.ones((batch_size * 2, 30, 30, 1))

        results = wrapper.predict(input_data, batch_size=batch_size)
        np.testing.assert_array_equal(input_data, results)

        # no batch size
        results = wrapper.predict(input_data)
        np.testing.assert_array_equal(input_data, results)

        # multiple inputs
        metadata = self._get_metadata() * 2
        wrapper = grpc_clients.GrpcModelWrapper(None, metadata)
        mocker.patch.object(wrapper, 'send_grpc', mock_send_grpc)
        input_data = [np.ones((batch_size * 2, 30, 30, 1))] * 2
        results = wrapper.predict(input_data, batch_size=batch_size)
        np.testing.assert_array_equal(input_data, results)

        # dictionary input
        metadata = self._get_metadata()
        wrapper = grpc_clients.GrpcModelWrapper(None, metadata)
        mocker.patch.object(wrapper, 'send_grpc', mock_send_grpc)
        input_data = {
            m['in_tensor_name']: np.ones((batch_size * 2, 30, 30, 1))
            for m in metadata
        }
        results = wrapper.predict(input_data, batch_size=batch_size)
        for m in metadata:
            np.testing.assert_array_equal(
                input_data[m['in_tensor_name']], results)
