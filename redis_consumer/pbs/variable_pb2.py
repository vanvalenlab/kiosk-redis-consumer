# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: variable.proto

from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='variable.proto',
  package='tensorflow',
  syntax='proto3',
  serialized_options=b'\n\030org.tensorflow.frameworkB\016VariableProtosP\001Z=github.com/tensorflow/tensorflow/tensorflow/go/core/framework\370\001\001',
  serialized_pb=b'\n\x0evariable.proto\x12\ntensorflow\"\xc8\x02\n\x0bVariableDef\x12\x15\n\rvariable_name\x18\x01 \x01(\t\x12\x1a\n\x12initial_value_name\x18\x06 \x01(\t\x12\x18\n\x10initializer_name\x18\x02 \x01(\t\x12\x15\n\rsnapshot_name\x18\x03 \x01(\t\x12\x39\n\x13save_slice_info_def\x18\x04 \x01(\x0b\x32\x1c.tensorflow.SaveSliceInfoDef\x12\x13\n\x0bis_resource\x18\x05 \x01(\x08\x12\x11\n\ttrainable\x18\x07 \x01(\x08\x12<\n\x0fsynchronization\x18\x08 \x01(\x0e\x32#.tensorflow.VariableSynchronization\x12\x34\n\x0b\x61ggregation\x18\t \x01(\x0e\x32\x1f.tensorflow.VariableAggregation\"`\n\x10SaveSliceInfoDef\x12\x11\n\tfull_name\x18\x01 \x01(\t\x12\x12\n\nfull_shape\x18\x02 \x03(\x03\x12\x12\n\nvar_offset\x18\x03 \x03(\x03\x12\x11\n\tvar_shape\x18\x04 \x03(\x03*\xac\x01\n\x17VariableSynchronization\x12!\n\x1dVARIABLE_SYNCHRONIZATION_AUTO\x10\x00\x12!\n\x1dVARIABLE_SYNCHRONIZATION_NONE\x10\x01\x12%\n!VARIABLE_SYNCHRONIZATION_ON_WRITE\x10\x02\x12$\n VARIABLE_SYNCHRONIZATION_ON_READ\x10\x03*\x9e\x01\n\x13VariableAggregation\x12\x1d\n\x19VARIABLE_AGGREGATION_NONE\x10\x00\x12\x1c\n\x18VARIABLE_AGGREGATION_SUM\x10\x01\x12\x1d\n\x19VARIABLE_AGGREGATION_MEAN\x10\x02\x12+\n\'VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA\x10\x03\x42n\n\x18org.tensorflow.frameworkB\x0eVariableProtosP\x01Z=github.com/tensorflow/tensorflow/tensorflow/go/core/framework\xf8\x01\x01\x62\x06proto3'
)

_VARIABLESYNCHRONIZATION = _descriptor.EnumDescriptor(
  name='VariableSynchronization',
  full_name='tensorflow.VariableSynchronization',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='VARIABLE_SYNCHRONIZATION_AUTO', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VARIABLE_SYNCHRONIZATION_NONE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VARIABLE_SYNCHRONIZATION_ON_WRITE', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VARIABLE_SYNCHRONIZATION_ON_READ', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=460,
  serialized_end=632,
)
_sym_db.RegisterEnumDescriptor(_VARIABLESYNCHRONIZATION)

VariableSynchronization = enum_type_wrapper.EnumTypeWrapper(_VARIABLESYNCHRONIZATION)
_VARIABLEAGGREGATION = _descriptor.EnumDescriptor(
  name='VariableAggregation',
  full_name='tensorflow.VariableAggregation',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='VARIABLE_AGGREGATION_NONE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VARIABLE_AGGREGATION_SUM', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VARIABLE_AGGREGATION_MEAN', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=635,
  serialized_end=793,
)
_sym_db.RegisterEnumDescriptor(_VARIABLEAGGREGATION)

VariableAggregation = enum_type_wrapper.EnumTypeWrapper(_VARIABLEAGGREGATION)
VARIABLE_SYNCHRONIZATION_AUTO = 0
VARIABLE_SYNCHRONIZATION_NONE = 1
VARIABLE_SYNCHRONIZATION_ON_WRITE = 2
VARIABLE_SYNCHRONIZATION_ON_READ = 3
VARIABLE_AGGREGATION_NONE = 0
VARIABLE_AGGREGATION_SUM = 1
VARIABLE_AGGREGATION_MEAN = 2
VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA = 3



_VARIABLEDEF = _descriptor.Descriptor(
  name='VariableDef',
  full_name='tensorflow.VariableDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='variable_name', full_name='tensorflow.VariableDef.variable_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='initial_value_name', full_name='tensorflow.VariableDef.initial_value_name', index=1,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='initializer_name', full_name='tensorflow.VariableDef.initializer_name', index=2,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='snapshot_name', full_name='tensorflow.VariableDef.snapshot_name', index=3,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='save_slice_info_def', full_name='tensorflow.VariableDef.save_slice_info_def', index=4,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='is_resource', full_name='tensorflow.VariableDef.is_resource', index=5,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='trainable', full_name='tensorflow.VariableDef.trainable', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='synchronization', full_name='tensorflow.VariableDef.synchronization', index=7,
      number=8, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='aggregation', full_name='tensorflow.VariableDef.aggregation', index=8,
      number=9, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=31,
  serialized_end=359,
)


_SAVESLICEINFODEF = _descriptor.Descriptor(
  name='SaveSliceInfoDef',
  full_name='tensorflow.SaveSliceInfoDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='full_name', full_name='tensorflow.SaveSliceInfoDef.full_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='full_shape', full_name='tensorflow.SaveSliceInfoDef.full_shape', index=1,
      number=2, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='var_offset', full_name='tensorflow.SaveSliceInfoDef.var_offset', index=2,
      number=3, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='var_shape', full_name='tensorflow.SaveSliceInfoDef.var_shape', index=3,
      number=4, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=361,
  serialized_end=457,
)

_VARIABLEDEF.fields_by_name['save_slice_info_def'].message_type = _SAVESLICEINFODEF
_VARIABLEDEF.fields_by_name['synchronization'].enum_type = _VARIABLESYNCHRONIZATION
_VARIABLEDEF.fields_by_name['aggregation'].enum_type = _VARIABLEAGGREGATION
DESCRIPTOR.message_types_by_name['VariableDef'] = _VARIABLEDEF
DESCRIPTOR.message_types_by_name['SaveSliceInfoDef'] = _SAVESLICEINFODEF
DESCRIPTOR.enum_types_by_name['VariableSynchronization'] = _VARIABLESYNCHRONIZATION
DESCRIPTOR.enum_types_by_name['VariableAggregation'] = _VARIABLEAGGREGATION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

VariableDef = _reflection.GeneratedProtocolMessageType('VariableDef', (_message.Message,), {
  'DESCRIPTOR' : _VARIABLEDEF,
  '__module__' : 'variable_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.VariableDef)
  })
_sym_db.RegisterMessage(VariableDef)

SaveSliceInfoDef = _reflection.GeneratedProtocolMessageType('SaveSliceInfoDef', (_message.Message,), {
  'DESCRIPTOR' : _SAVESLICEINFODEF,
  '__module__' : 'variable_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.SaveSliceInfoDef)
  })
_sym_db.RegisterMessage(SaveSliceInfoDef)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
