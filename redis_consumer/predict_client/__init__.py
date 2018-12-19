# Copied from https://github.com/epigramai/tfserving-python-predict-client
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from redis_consumer.predict_client import pbs
from redis_consumer.predict_client import dict_to_protobuf
from redis_consumer.predict_client import grpc_client
from redis_consumer.predict_client import util
from redis_consumer.predict_client.grpc_client import GrpcClient

del absolute_import
del division
del print_function
