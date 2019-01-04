# Copied from https://github.com/epigramai/tfserving-python-predict-client
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time

import grpc
from grpc import RpcError

import grpc.beta.implementations
from grpc._cython import cygrpc

from redis_consumer.pbs.prediction_service_pb2_grpc import PredictionServiceStub
from redis_consumer.pbs.processing_service_pb2_grpc import ProcessingServiceStub
from redis_consumer.pbs.predict_pb2 import PredictRequest
from redis_consumer.pbs.process_pb2 import ProcessRequest
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
        t = time.time()
        channel = grpc.insecure_channel(
            target=self.host,
            options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                     (cygrpc.ChannelArgKey.max_receive_message_length, -1)])
        self.logger.debug('Establishing insecure channel took: %s',
                          time.time() - t)
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
        self.logger.info('Sending request to tensorflow-serving model.')
        self.logger.info('Host: %s', self.host)
        self.logger.info('Model name: %s', self.model_name)
        self.logger.info('Model version: %s', self.model_version)

        # Create gRPC client and request
        channel = self.insecure_channel()

        t = time.time()
        stub = PredictionServiceStub(channel)
        self.logger.debug('Creating stub took: %s', time.time() - t)

        t = time.time()
        request = PredictRequest()
        self.logger.debug('Creating request object took: %s', time.time() - t)

        request.model_spec.name = self.model_name

        if self.model_version > 0:
            request.model_spec.version.value = self.model_version

        t = time.time()
        for d in request_data:
            tensor_proto = make_tensor_proto(d['data'], d['in_tensor_dtype'])
            request.inputs[d['in_tensor_name']].CopyFrom(tensor_proto)

        self.logger.debug('Making tensor protos took: %s', time.time() - t)

        try:
            t = time.time()
            predict_response = stub.Predict(request, timeout=request_timeout)

            self.logger.debug('Actual request took: %ss', time.time() - t)

            predict_response_dict = grpc_response_to_dict(predict_response)

            keys = [k for k in predict_response_dict]
            self.logger.info('Got predict_response with keys: %s', keys)

            return predict_response_dict

        except RpcError as e:
            self.logger.error(e)
            self.logger.error('Prediction failed!')
            raise e

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

        t = time.time()
        stub = ProcessingServiceStub(channel)
        self.logger.debug('Creating stub took %ss', time.time() - t)

        t = time.time()
        request = ProcessRequest()
        self.logger.debug('Creating request object took: %s', time.time() - t)

        request.function_spec.name = self.function_name
        request.function_spec.type = self.process_type

        t = time.time()
        for d in request_data:
            tensor_proto = make_tensor_proto(d['data'], d['in_tensor_dtype'])
            request.inputs[d['in_tensor_name']].CopyFrom(tensor_proto)

        self.logger.debug('Making tensor protos took: %s', time.time() - t)

        try:
            t = time.time()
            response = stub.Process(request, timeout=request_timeout)

            self.logger.debug('Actual request took: %ss', time.time() - t)

            response_dict = grpc_response_to_dict(response)

            keys = [k for k in response_dict]
            self.logger.info('Got processing_response with keys: %s', keys)

            return response_dict

        except RpcError as e:
            self.logger.error(e)
            self.logger.error('Prediction failed!')
            raise e

        return {}
