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
"""GRPC Clients inspired by
https://github.com/epigramai/tfserving-python-predict-client
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import time
import timeit
import six

import dict_to_protobuf
from google.protobuf.json_format import MessageToJson
import grpc
import grpc.beta.implementations
from grpc._cython import cygrpc
import numpy as np
from tensorflow.core.framework.types_pb2 import DESCRIPTOR
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis.get_model_metadata_pb2 import GetModelMetadataRequest

from redis_consumer import settings


logger = logging.getLogger('redis_consumer.grpc_clients')


dtype_to_number = {
    i.name: i.number for i in DESCRIPTOR.enum_types_by_name['DataType'].values
}

# TODO: build this dynamically
number_to_dtype_value = {
    1: 'float_val',
    2: 'double_val',
    3: 'int_val',
    4: 'int_val',
    5: 'int_val',
    6: 'int_val',
    7: 'string_val',
    8: 'scomplex_val',
    9: 'int64_val',
    10: 'bool_val',
    18: 'dcomplex_val',
    19: 'half_val',
    20: 'resource_handle_val'
}


def grpc_response_to_dict(grpc_response):
    # TODO: 'unicode' object has no attribute 'ListFields'
    # response_dict = dict_to_protobuf.protobuf_to_dict(grpc_response)
    # return response_dict
    grpc_response_dict = dict()

    for k in grpc_response.outputs:
        shape = [x.size for x in grpc_response.outputs[k].tensor_shape.dim]

        dtype_constant = grpc_response.outputs[k].dtype

        if dtype_constant not in number_to_dtype_value:
            grpc_response_dict[k] = 'value not found'
            logger.error('Tensor output data type not supported. '
                         'Returning empty dict.')

        dt = number_to_dtype_value[dtype_constant]
        if shape == [1]:
            grpc_response_dict[k] = eval(
                'grpc_response.outputs[k].' + dt)[0]
        else:
            grpc_response_dict[k] = np.array(
                eval('grpc_response.outputs[k].' + dt)).reshape(shape)

    return grpc_response_dict


def make_tensor_proto(data, dtype):
    tensor_proto = TensorProto()

    if isinstance(dtype, six.string_types):
        dtype = dtype_to_number[dtype]

    dim = [{'size': 1}]
    values = [data]

    if hasattr(data, 'shape'):
        dim = [{'size': dim} for dim in data.shape]
        values = list(data.reshape(-1))

    tensor_proto_dict = {
        'dtype': dtype,
        'tensor_shape': {
            'dim': dim
        },
        number_to_dtype_value[dtype]: values
    }
    dict_to_protobuf.dict_to_protobuf(tensor_proto_dict, tensor_proto)

    return tensor_proto


class GrpcClient(object):
    """Abstract class for all gRPC clients.

    Arguments:
        host: string, the hostname and port of the server (`localhost:8080`)
    """

    def __init__(self, host):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.host = host
        self.options = [
            (cygrpc.ChannelArgKey.max_send_message_length, -1),
            (cygrpc.ChannelArgKey.max_receive_message_length, -1),
            ('grpc.default_compression_algorithm', cygrpc.CompressionAlgorithm.gzip),
            ('grpc.grpc.default_compression_level', cygrpc.CompressionLevel.high)
        ]

    def insecure_channel(self):
        """Create an insecure channel with max message length.

        Returns:
            channel: grpc.insecure channel object
        """
        t = timeit.default_timer()
        channel = grpc.insecure_channel(target=self.host, options=self.options)
        self.logger.debug('Establishing insecure channel took: %s',
                          timeit.default_timer() - t)
        return channel


class PredictClient(GrpcClient):
    """gRPC Client for tensorflow-serving API.

    Arguments:
        host: string, the hostname and port of the server (`localhost:8080`)
        model_name: string, name of model served by tensorflow-serving
        model_version: integer, version of the named model
    """

    def __init__(self, host, model_name, model_version):
        super(PredictClient, self).__init__(host)
        self.logger = logging.getLogger('gRPC:{}:{}'.format(
            model_name, model_version
        ))
        self.model_name = model_name
        self.model_version = model_version

        self.stub_lookup = {
            GetModelMetadataRequest: 'GetModelMetadata',
            PredictRequest: 'Predict',
        }

        # Retry-able gRPC status codes
        self.retry_status_codes = {
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            grpc.StatusCode.UNAVAILABLE
        }

    def _retry_grpc(self, request, request_timeout):
        request_name = request.__class__.__name__
        self.logger.info('Sending %s to %s.', request_name, self.host)

        true_failures, count = 0, 0

        retrying = True
        while retrying:
            with self.insecure_channel() as channel:
                # pylint: disable=E1101
                try:
                    t = timeit.default_timer()

                    stub = PredictionServiceStub(channel)

                    api_endpoint_name = self.stub_lookup.get(request.__class__)
                    api_call = getattr(stub, api_endpoint_name)
                    response = api_call(request, timeout=request_timeout)

                    self.logger.debug('%s finished in %s seconds (%s retries).',
                                      request_name, timeit.default_timer() - t,
                                      true_failures)
                    return response

                except grpc.RpcError as err:
                    if true_failures > settings.MAX_RETRY > 0:
                        retrying = False
                        self.logger.error('%s has failed %s times due to err '
                                          '%s', request_name, count, err)
                        raise err

                    if err.code() in self.retry_status_codes:
                        count += 1
                        is_true_failure = err.code() != grpc.StatusCode.UNAVAILABLE
                        true_failures += int(is_true_failure)

                        self.logger.warning('%sException `%s: %s` during '
                                            '%s %s to model %s:%s. Waiting %s '
                                            'seconds before retrying.',
                                            type(err).__name__,
                                            err.code().name, err.details(),
                                            self.__class__.__name__,
                                            request_name,
                                            self.model_name, self.model_version,
                                            settings.GRPC_BACKOFF)

                        time.sleep(settings.GRPC_BACKOFF)  # sleep before retry
                        retrying = True  # Unneccessary but explicit
                    else:
                        retrying = False
                        raise err
                except Exception as err:
                    retrying = False
                    self.logger.error('Encountered %s during %s to model '
                                      '%s:%s: %s', type(err).__name__,
                                      request_name, self.model_name,
                                      self.model_version, err)
                    raise err

    def predict(self, request_data, request_timeout=10):
        # pylint: disable=E1101
        request = PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.version.value = self.model_version

        for d in request_data:
            tensor_proto = make_tensor_proto(d['data'], d['in_tensor_dtype'])
            request.inputs[d['in_tensor_name']].CopyFrom(tensor_proto)

        response = self._retry_grpc(request, request_timeout)
        response_dict = grpc_response_to_dict(response)

        self.logger.info('Got PredictResponse with keys: %s.',
                         list(response_dict))

        return response_dict

    def get_model_metadata(self, request_timeout=10):
        # pylint: disable=E1101
        request = GetModelMetadataRequest()
        request.metadata_field.append('signature_def')
        request.model_spec.name = self.model_name
        request.model_spec.version.value = self.model_version

        response = self._retry_grpc(request, request_timeout)
        response_dict = json.loads(MessageToJson(response))
        return response_dict


class GrpcModelWrapper(object):
    """A wrapper class that mocks a Keras model using a gRPC client.

    https://github.com/vanvalenlab/deepcell-tf/blob/master/deepcell/applications
    """

    def __init__(self, client, model_metadata):
        self._client = client
        self._metadata = model_metadata

        shapes = [
            tuple([int(x) for x in m['in_tensor_shape'].split(',')])
            for m in self._metadata
        ]
        if len(shapes) == 1:
            shapes = shapes[0]
        self.input_shape = shapes

    def send_grpc(self, img):
        """Use the TensorFlow Serving gRPC API for model inference on an image.

        Args:
            img (numpy.array): The image to send to the model

        Returns:
            numpy.array: The results of model inference.
        """
        start = timeit.default_timer()

        # cast input as list
        if not isinstance(img, list):
            img = [img]

        if len(self._metadata) != len(img):
            raise ValueError('Expected {} model inputs but got {}.'.format(
                len(self._metadata), len(img)))

        req_data = []

        for i, m in enumerate(self._metadata):
            data = img[i]

            if m['in_tensor_dtype'] == 'DT_HALF':
                # seems like should cast to "half"
                # but the model rejects the type, wants "int" or "long"
                data = data.astype('int')

            req_data.append({
                'in_tensor_name': m['in_tensor_name'],
                'in_tensor_dtype': m['in_tensor_dtype'],
                'data': data
            })

        self._client.logger.debug('Predicting...')
        prediction = self._client.predict(req_data, settings.GRPC_TIMEOUT)
        results = [prediction[k] for k in sorted(prediction.keys())]

        self._client.logger.debug('Got prediction results of shape %s in %s s.',
                                  [r.shape for r in results],
                                  timeit.default_timer() - start)

        return results

    def get_batch_size(self):
        """Calculate the best batch size based on TF_MAX_BATCH_SIZE and
        TF_MIN_MODEL_SIZE
        """
        input_shape = self.input_shape
        if not isinstance(input_shape, list):
            input_shape = [input_shape]

        ratio = 1
        for shape in input_shape:
            rank = len(shape)
            ratio *= (shape[rank - 3] / settings.TF_MIN_MODEL_SIZE) * \
                     (shape[rank - 2] / settings.TF_MIN_MODEL_SIZE) * \
                     (shape[rank - 1])

        batch_size = int(settings.TF_MAX_BATCH_SIZE // ratio)
        return batch_size

    def predict(self, tiles, batch_size=None):
        # TODO: Can the result size be known beforehand via model metadata?
        results = []

        if isinstance(tiles, dict):
            tiles = [tiles[m['in_tensor_name']] for m in self._metadata]

        if not isinstance(tiles, list):
            tiles = [tiles]

        # if batch_size is None:
        #     batch_size = self.get_batch_size()

        batch_size = 16

        for t in range(0, tiles[0].shape[0], batch_size):
            inputs = [tile[t:t + batch_size] for tile in tiles]
            inputs = inputs[0] if len(inputs) == 1 else inputs
            output = self.send_grpc(inputs)

            if len(results) == 0:
                results = output
            else:
                for i, o in enumerate(output):
                    results[i] = np.vstack((results[i], o))

        return results[0] if len(results) == 1 else results
