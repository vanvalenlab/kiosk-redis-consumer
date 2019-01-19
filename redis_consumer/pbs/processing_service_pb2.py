# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: processing_service.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import redis_consumer.pbs.process_pb2 as process__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='processing_service.proto',
  package='tensorflow.serving',
  syntax='proto3',
  serialized_options=_b('\370\001\001'),
  serialized_pb=_b('\n\x18processing_service.proto\x12\x12tensorflow.serving\x1a\rprocess.proto2\xd3\x01\n\x11ProcessingService\x12R\n\x07Process\x12\".tensorflow.serving.ProcessRequest\x1a#.tensorflow.serving.ProcessResponse\x12j\n\rStreamProcess\x12).tensorflow.serving.ChunkedProcessRequest\x1a*.tensorflow.serving.ChunkedProcessResponse(\x01\x30\x01\x42\x03\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[process__pb2.DESCRIPTOR,])



_sym_db.RegisterFileDescriptor(DESCRIPTOR)


DESCRIPTOR._options = None

_PROCESSINGSERVICE = _descriptor.ServiceDescriptor(
  name='ProcessingService',
  full_name='tensorflow.serving.ProcessingService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=64,
  serialized_end=275,
  methods=[
  _descriptor.MethodDescriptor(
    name='Process',
    full_name='tensorflow.serving.ProcessingService.Process',
    index=0,
    containing_service=None,
    input_type=process__pb2._PROCESSREQUEST,
    output_type=process__pb2._PROCESSRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='StreamProcess',
    full_name='tensorflow.serving.ProcessingService.StreamProcess',
    index=1,
    containing_service=None,
    input_type=process__pb2._CHUNKEDPROCESSREQUEST,
    output_type=process__pb2._CHUNKEDPROCESSRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_PROCESSINGSERVICE)

DESCRIPTOR.services_by_name['ProcessingService'] = _PROCESSINGSERVICE

# @@protoc_insertion_point(module_scope)
