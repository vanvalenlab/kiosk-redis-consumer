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

from redis_consumer.predict_client import pbs
from redis_consumer.predict_client.util import predict_response_to_dict
from redis_consumer.predict_client.util import make_tensor_proto


class GrpcClient:
    def __init__(self, host, model_name, model_version):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.host = host
        self.model_name = model_name
        self.model_version = model_version

    def insecure_channel(self):
        return grpc.insecure_channel(
            target=self.host,
            options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                     (cygrpc.ChannelArgKey.max_receive_message_length, -1)])

    def predict(self, request_data, request_timeout=10):
        self.logger.info('Sending request to tfserving model')
        self.logger.info('Host: %s', self.host)
        self.logger.info('Model name: %s', self.model_name)
        self.logger.info('Model version: %s', self.model_version)

        # Create gRPC client and request
        t = time.time()
        channel = self.insecure_channel()
        self.logger.debug('Establishing insecure channel took: %s',
                          time.time() - t)

        t = time.time()
        stub = pbs.prediction_service_pb2_grpc.PredictionServiceStub(channel)
        self.logger.debug('Creating stub took: %s', time.time() - t)

        t = time.time()
        request = pbs.predict_pb2.PredictRequest()
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

            self.logger.debug('Actual request took: %s seconds',
                              time.time() - t)

            predict_response_dict = predict_response_to_dict(predict_response)

            keys = [k for k in predict_response_dict]
            self.logger.info('Got predict_response with keys: %s', keys)

            return predict_response_dict

        except RpcError as e:
            self.logger.error(e)
            self.logger.error('Prediction failed!')
            raise e

        return {}
