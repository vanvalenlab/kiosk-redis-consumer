# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: prediction_service.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import redis_consumer.pbs.get_model_metadata_pb2 as get__model__metadata__pb2
import redis_consumer.pbs.predict_pb2 as predict__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='prediction_service.proto',
  package='tensorflow.serving',
  syntax='proto3',
  serialized_options=b'\370\001\001',
  serialized_pb=b'\n\x18prediction_service.proto\x12\x12tensorflow.serving\x1a\x18get_model_metadata.proto\x1a\rpredict.proto2\xd6\x01\n\x11PredictionService\x12R\n\x07Predict\x12\".tensorflow.serving.PredictRequest\x1a#.tensorflow.serving.PredictResponse\x12m\n\x10GetModelMetadata\x12+.tensorflow.serving.GetModelMetadataRequest\x1a,.tensorflow.serving.GetModelMetadataResponseB\x03\xf8\x01\x01\x62\x06proto3'
  ,
  dependencies=[get__model__metadata__pb2.DESCRIPTOR,predict__pb2.DESCRIPTOR,])



_sym_db.RegisterFileDescriptor(DESCRIPTOR)


DESCRIPTOR._options = None

_PREDICTIONSERVICE = _descriptor.ServiceDescriptor(
  name='PredictionService',
  full_name='tensorflow.serving.PredictionService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=90,
  serialized_end=304,
  methods=[
  _descriptor.MethodDescriptor(
    name='Predict',
    full_name='tensorflow.serving.PredictionService.Predict',
    index=0,
    containing_service=None,
    input_type=predict__pb2._PREDICTREQUEST,
    output_type=predict__pb2._PREDICTRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetModelMetadata',
    full_name='tensorflow.serving.PredictionService.GetModelMetadata',
    index=1,
    containing_service=None,
    input_type=get__model__metadata__pb2._GETMODELMETADATAREQUEST,
    output_type=get__model__metadata__pb2._GETMODELMETADATARESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_PREDICTIONSERVICE)

DESCRIPTOR.services_by_name['PredictionService'] = _PREDICTIONSERVICE

# @@protoc_insertion_point(module_scope)
