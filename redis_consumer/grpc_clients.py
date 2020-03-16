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

import logging
import time
import timeit

import grpc
from grpc import RpcError
import grpc.beta.implementations
from grpc._cython import cygrpc

import numpy as np

from redis_consumer.pbs.prediction_service_pb2_grpc import PredictionServiceStub
from redis_consumer.pbs.processing_service_pb2_grpc import ProcessingServiceStub
from redis_consumer.pbs.predict_pb2 import PredictRequest
from redis_consumer.pbs.process_pb2 import ProcessRequest
from redis_consumer.pbs.process_pb2 import ChunkedProcessRequest
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

    def insecure_channel(self):
        """Create an insecure channel with max message length.

        Returns:
            channel: grpc.insecure channel object
        """
        t = timeit.default_timer()
        options = [
            (cygrpc.ChannelArgKey.max_send_message_length, -1),
            (cygrpc.ChannelArgKey.max_receive_message_length, -1),
            ('grpc.default_compression_algorithm', cygrpc.CompressionAlgorithm.gzip),
            ('grpc.grpc.default_compression_level', cygrpc.CompressionLevel.high)
        ]
        channel = grpc.insecure_channel(target=self.host, options=options)
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

    def predict(self, request_data, request_timeout=10):
        self.logger.info('Sending request to %s model %s:%s.',
                         self.host, self.model_name, self.model_version)

        channel = self.insecure_channel()

        t = timeit.default_timer()
        stub = PredictionServiceStub(channel)
        self.logger.debug('Created TensorFlowServingServiceStub in %s seconds.',
                          timeit.default_timer() - t)

        t = timeit.default_timer()
        request = PredictRequest()
        self.logger.debug('Created TensorFlowServingRequest object in %s '
                          'seconds.', timeit.default_timer() - t)

        request.model_spec.name = self.model_name  # pylint: disable=E1101

        if self.model_version > 0:
            # pylint: disable=E1101
            request.model_spec.version.value = self.model_version

        t = timeit.default_timer()
        for d in request_data:
            tensor_proto = make_tensor_proto(d['data'], d['in_tensor_dtype'])
            # pylint: disable=E1101
            request.inputs[d['in_tensor_name']].CopyFrom(tensor_proto)

        self.logger.debug('Made tensor protos in %s seconds.',
                          timeit.default_timer() - t)

        try:
            t = timeit.default_timer()
            predict_response = stub.Predict(request, timeout=request_timeout)
            self.logger.debug('gRPC TensorFlowServingRequest finished in %s '
                              'seconds.', timeit.default_timer() - t)

            t = timeit.default_timer()
            predict_response_dict = grpc_response_to_dict(predict_response)
            self.logger.debug('gRPC TensorFlowServingProtobufConversion took '
                              '%s seconds.', timeit.default_timer() - t)

            keys = [k for k in predict_response_dict]
            self.logger.info('Got TensorFlowServingResponse with keys: %s ',
                             keys)
            channel.close()
            return predict_response_dict

        except RpcError as err:
            self.logger.error('Prediction failed due to: %s', err)
            channel.close()
            raise err

        channel.close()
        return {}


class ProcessClient(GrpcClient):
    """gRPC Client for data-processing API.

    Arguments:
        host: string, the hostname and port of the server (`localhost:8080`)
        process_type: string, pre or post processing
        function_name: string, name of processing function
    """

    def __init__(self, host, process_type, function_name):
        super(ProcessClient, self).__init__(host)
        self.process_type = process_type
        self.function_name = function_name

    def process(self, request_data, request_timeout=10):
        self.logger.info('Sending request to %s %s-process data with the '
                         'data-processing API at %s.', self.function_name,
                         self.process_type, self.host)

        # Create gRPC client and request
        channel = self.insecure_channel()

        t = timeit.default_timer()
        stub = ProcessingServiceStub(channel)
        self.logger.debug('Created DataProcessingProcessingServiceStub in %s '
                          'seconds.', timeit.default_timer() - t)

        t = timeit.default_timer()
        request = ProcessRequest()
        self.logger.debug('Created DataProcessingRequest object in %s seconds.',
                          timeit.default_timer() - t)

        # pylint: disable=E1101
        request.function_spec.name = self.function_name
        request.function_spec.type = self.process_type
        # pylint: enable=E1101

        t = timeit.default_timer()
        for d in request_data:
            tensor_proto = make_tensor_proto(d['data'], d['in_tensor_dtype'])
            # pylint: disable=E1101
            request.inputs[d['in_tensor_name']].CopyFrom(tensor_proto)

        self.logger.debug('Made tensor protos in %s seconds.',
                          timeit.default_timer() - t)

        try:
            t = timeit.default_timer()
            response = stub.Process(request, timeout=request_timeout)
            self.logger.debug('gRPC DataProcessingRequest finished in %s '
                              'seconds.', timeit.default_timer() - t)

            t = timeit.default_timer()
            response_dict = grpc_response_to_dict(response)
            self.logger.debug('gRPC DataProcessingProtobufConversion took %s '
                              'seconds.', timeit.default_timer() - t)

            keys = [k for k in response_dict]
            self.logger.debug('Got processing_response with keys: %s', keys)
            channel.close()
            return response_dict

        except RpcError as err:
            self.logger.error('Processing failed due to: %s', err)
            channel.close()
            raise err

        channel.close()
        return {}

    def stream_process(self, request_data, request_timeout=10):
        self.logger.info('Sending request to %s %s-process data with the '
                         'data-processing API at %s.', self.function_name,
                         self.process_type, self.host)

        # Create gRPC client and request
        channel = self.insecure_channel()

        t = timeit.default_timer()
        stub = ProcessingServiceStub(channel)
        self.logger.debug('Created stub in %s seconds.',
                          timeit.default_timer() - t)
        chunk_size = 64 * 1024  # 64 kB is recommended payload size

        def request_iterator(image):
            dtype = str(image.dtype)
            shape = list(image.shape)
            bytearr = image.tobytes()

            self.logger.info('Streaming %s bytes in %s requests',
                             len(bytearr), chunk_size % len(bytearr))

            for i in range(0, len(bytearr), chunk_size):
                request = ChunkedProcessRequest()
                # pylint: disable=E1101
                request.function_spec.name = self.function_name
                request.function_spec.type = self.process_type
                request.shape[:] = shape
                request.dtype = dtype
                request.inputs['data'] = bytearr[i: i + chunk_size]
                # pylint: enable=E1101
                yield request

        try:
            t = timeit.default_timer()
            req_iter = request_iterator(request_data[0]['data'])
            res_iter = stub.StreamProcess(req_iter, timeout=request_timeout)

            shape = None
            dtype = None
            processed_bytes = []
            for response in res_iter:
                shape = tuple(response.shape)
                dtype = str(response.dtype)
                processed_bytes.append(response.outputs['data'])

            npbytes = b''.join(processed_bytes)
            # Got response stream of %s bytes in %s seconds.
            self.logger.info('gRPC DataProcessingStreamRequest of %s bytes '
                             'finished in %s seconds.', len(npbytes),
                             timeit.default_timer() - t)

            t = timeit.default_timer()
            processed_image = np.frombuffer(npbytes, dtype=dtype)
            results = processed_image.reshape(shape)
            self.logger.info('gRPC DataProcessingStreamConversion from %s bytes'
                             ' to a numpy array of shape %s in %s seconds.',
                             len(npbytes), results.shape,
                             timeit.default_timer() - t)
            channel.close()
            return {'results': results}

        except RpcError as err:
            self.logger.error('Processing failed due to: %s', err)
            channel.close()
            raise err

        channel.close()
        return {}


class TrackingClient(GrpcClient):
    """gRPC Client for tensorflow-serving API.

    Arguments:
        host: string, the hostname and port of the server (`localhost:8080`)
        model_name: string, name of model served by tensorflow-serving
        model_version: integer, version of the named model
    """

    def __init__(self, host, redis_hash, model_name, model_version, progress_callback):
        super(TrackingClient, self).__init__(host)
        self.redis_hash = redis_hash
        self.model_name = model_name
        self.model_version = model_version
        self.progress_callback = progress_callback

    def _single_request(self, stub, request, request_timeout=100):
        while True:
            try:
                predict_response = stub.Predict(request, timeout=request_timeout)
                predict_response_dict = grpc_response_to_dict(predict_response)
                keys = [k for k in predict_response_dict]
                return predict_response_dict['prediction']

            except RpcError as err:
                self.logger.error(err)
                self.logger.error('Single prediction failed! Trying again...')
                time.sleep(10)

    def _predict(self, data, request_timeout=100):
        self.logger.info('Sending tracking prediction to %s model %s:%s.',
                         self.host, self.model_name, self.model_version)

        channel = self.insecure_channel()
        stub = PredictionServiceStub(channel)

        t = timeit.default_timer()
        num_preds = data[0].shape[0]

        predictions = []
        for data_i in range(num_preds):
            request = PredictRequest()
            request.model_spec.name = self.model_name  # pylint: disable=E1101

            if self.model_version > 0:
                request.model_spec.version.value = self.model_version

            for i, model_input in enumerate(data):
                model_input = np.expand_dims(model_input[data_i], axis=0)
                tensor_proto = make_tensor_proto(model_input, 'DT_FLOAT')
                request.inputs["input{}".format(i)].CopyFrom(tensor_proto)

            # Select only last dimension in order to drop batch axis
            predictions.append(self._single_request(stub, request)[-1])

        self.logger.info('Predicting everything took: %s seconds',
                         timeit.default_timer() - t)
        channel.close()
        return predictions

    def predict(self, data):
        self.logger.info("Got data with shape %s",
                         [model_input.shape for model_input in data])

        return np.array(self._predict(data))

    def progress(self, progress):
        """
        Update the internal state regarding progress

        Arguments:
            progress: float, the progress in the interval [0, 1]
        """
        progress *= 100
        # clamp to an integer between 0 and 100
        progress = min(100, max(0, round(progress)))
        self.progress_callback(self.redis_hash, progress)
