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
"""Tests for gRPC Clients"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow_serving.apis.predict_pb2 import PredictResponse

from redis_consumer.testing_utils import _get_image

from redis_consumer import grpc_clients


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