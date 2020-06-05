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

import grpc
from grpc import RpcError
import grpc.beta.implementations
from grpc._cython import cygrpc

import numpy as np

from google.protobuf.json_format import MessageToJson

from redis_consumer import settings
from redis_consumer.pbs.prediction_service_pb2_grpc import PredictionServiceStub
from redis_consumer.pbs.predict_pb2 import PredictRequest
from redis_consumer.pbs.get_model_metadata_pb2 import GetModelMetadataRequest
from redis_consumer.utils import grpc_response_to_dict
from redis_consumer.utils import make_tensor_proto


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
        self.model_name = model_name
        self.model_version = model_version

        self.stub_lookup = {
            GetModelMetadataRequest: 'GetModelMetadata',
            PredictRequest: 'Predict',
        }

    def _retry_grpc(self, request, request_timeout):
        request_name = request.__class__.__name__
        self.logger.info('Sending %s to %s model %s:%s.',
                         request_name, self.host,
                         self.model_name, self.model_version)

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

                    self.logger.debug('%s finished in %s seconds.',
                                      request_name, timeit.default_timer() - t)
                    return response

                except grpc.RpcError as err:
                    if true_failures > settings.MAX_RETRY > 0:
                        retrying = False
                        self.logger.error('%s has failed %s times due to err '
                                          '%s', request_name, count, err)
                        raise err

                    if err.code() in settings.GRPC_RETRY_STATUSES:
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
        self.logger.info('Sending PredictRequest to %s model %s:%s.',
                         self.host, self.model_name, self.model_version)

        t = timeit.default_timer()
        request = PredictRequest()
        self.logger.debug('Created PredictRequest object in %s seconds.',
                          timeit.default_timer() - t)

        # pylint: disable=E1101
        request.model_spec.name = self.model_name
        request.model_spec.version.value = self.model_version

        t = timeit.default_timer()
        for d in request_data:
            tensor_proto = make_tensor_proto(d['data'], d['in_tensor_dtype'])
            request.inputs[d['in_tensor_name']].CopyFrom(tensor_proto)

        self.logger.debug('Made tensor protos in %s seconds.',
                          timeit.default_timer() - t)

        response = self._retry_grpc(request, request_timeout)
        response_dict = grpc_response_to_dict(response)

        self.logger.info('Got PredictResponse with keys: %s ',
                         list(response_dict))

        return response_dict

    def get_model_metadata(self, request_timeout=10):
        self.logger.info('Sending GetModelMetadataRequest to %s model %s:%s.',
                         self.host, self.model_name, self.model_version)

        # pylint: disable=E1101
        request = GetModelMetadataRequest()
        request.metadata_field.append('signature_def')
        request.model_spec.name = self.model_name
        request.model_spec.version.value = self.model_version

        response = self._retry_grpc(request, request_timeout)

        t = timeit.default_timer()

        response_dict = json.loads(MessageToJson(response))

        self.logger.debug('gRPC GetModelMetadataProtobufConversion took '
                          '%s seconds.', timeit.default_timer() - t)

        return response_dict


class TrackingClient(PredictClient):
    """gRPC Client for tensorflow-serving API.

    Arguments:
        host: string, the hostname and port of the server (`localhost:8080`)
        model_name: string, name of model served by tensorflow-serving
        model_version: integer, version of the named model
    """

    def __init__(self, host, model_name, model_version,
                 redis_hash, progress_callback):
        self.redis_hash = redis_hash
        self.progress_callback = progress_callback
        super(TrackingClient, self).__init__(host, model_name, model_version)

    def predict(self, data, request_timeout=10):
        t = timeit.default_timer()
        self.logger.info('Tracking data of shape %s with %s model %s:%s.',
                         [d.shape for d in data],
                         self.host, self.model_name, self.model_version)

        batch_size = settings.TF_MAX_BATCH_SIZE
        results = []
        for frame in range(0, data[0].shape[0], batch_size):
            request_data = []
            for i, model_input in enumerate(data):
                d = {
                    'in_tensor_name': 'input{}'.format(i),
                    'in_tensor_dtype': 'DT_FLOAT',
                    'data': model_input[frame:frame + batch_size]
                }
                request_data.append(d)

            response_dict = super(TrackingClient, self).predict(
                request_data, request_timeout)

            output = response_dict['prediction']
            if len(results) == 0:
                results = output
            else:
                results = np.vstack((results, output))

        self.logger.info('Tracked %s input pairs in %s seconds.',
                         data[0].shape[0], timeit.default_timer() - t)

        return np.array(results)

    def progress(self, progress):
        """Update the internal state regarding progress

        Arguments:
            progress: float, the progress in the interval [0, 1]
        """
        progress *= 100
        # clamp to an integer between 0 and 100
        progress = min(100, max(0, round(progress)))
        self.progress_callback(self.redis_hash, progress)
