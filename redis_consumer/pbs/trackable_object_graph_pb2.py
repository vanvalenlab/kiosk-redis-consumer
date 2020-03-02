# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: trackable_object_graph.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='trackable_object_graph.proto',
  package='tensorflow',
  syntax='proto3',
  serialized_options=b'\370\001\001',
  serialized_pb=b'\n\x1ctrackable_object_graph.proto\x12\ntensorflow\"\x83\x05\n\x14TrackableObjectGraph\x12?\n\x05nodes\x18\x01 \x03(\x0b\x32\x30.tensorflow.TrackableObjectGraph.TrackableObject\x1a\xa9\x04\n\x0fTrackableObject\x12R\n\x08\x63hildren\x18\x01 \x03(\x0b\x32@.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference\x12U\n\nattributes\x18\x02 \x03(\x0b\x32\x41.tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor\x12^\n\x0eslot_variables\x18\x03 \x03(\x0b\x32\x46.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference\x1a\x36\n\x0fObjectReference\x12\x0f\n\x07node_id\x18\x01 \x01(\x05\x12\x12\n\nlocal_name\x18\x02 \x01(\t\x1a\x65\n\x10SerializedTensor\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x11\n\tfull_name\x18\x02 \x01(\t\x12\x16\n\x0e\x63heckpoint_key\x18\x03 \x01(\t\x12\x18\n\x10optional_restore\x18\x04 \x01(\x08\x1al\n\x15SlotVariableReference\x12!\n\x19original_variable_node_id\x18\x01 \x01(\x05\x12\x11\n\tslot_name\x18\x02 \x01(\t\x12\x1d\n\x15slot_variable_node_id\x18\x03 \x01(\x05\x42\x03\xf8\x01\x01\x62\x06proto3'
)




_TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT_OBJECTREFERENCE = _descriptor.Descriptor(
  name='ObjectReference',
  full_name='tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='node_id', full_name='tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.node_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='local_name', full_name='tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.local_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=421,
  serialized_end=475,
)

_TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT_SERIALIZEDTENSOR = _descriptor.Descriptor(
  name='SerializedTensor',
  full_name='tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='full_name', full_name='tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.full_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='checkpoint_key', full_name='tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.checkpoint_key', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='optional_restore', full_name='tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.optional_restore', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
  serialized_start=477,
  serialized_end=578,
)

_TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT_SLOTVARIABLEREFERENCE = _descriptor.Descriptor(
  name='SlotVariableReference',
  full_name='tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='original_variable_node_id', full_name='tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.original_variable_node_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='slot_name', full_name='tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.slot_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='slot_variable_node_id', full_name='tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.slot_variable_node_id', index=2,
      number=3, type=5, cpp_type=1, label=1,
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
  serialized_start=580,
  serialized_end=688,
)

_TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT = _descriptor.Descriptor(
  name='TrackableObject',
  full_name='tensorflow.TrackableObjectGraph.TrackableObject',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='children', full_name='tensorflow.TrackableObjectGraph.TrackableObject.children', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='attributes', full_name='tensorflow.TrackableObjectGraph.TrackableObject.attributes', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='slot_variables', full_name='tensorflow.TrackableObjectGraph.TrackableObject.slot_variables', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT_OBJECTREFERENCE, _TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT_SERIALIZEDTENSOR, _TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT_SLOTVARIABLEREFERENCE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=135,
  serialized_end=688,
)

_TRACKABLEOBJECTGRAPH = _descriptor.Descriptor(
  name='TrackableObjectGraph',
  full_name='tensorflow.TrackableObjectGraph',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='nodes', full_name='tensorflow.TrackableObjectGraph.nodes', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=45,
  serialized_end=688,
)

_TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT_OBJECTREFERENCE.containing_type = _TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT
_TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT_SERIALIZEDTENSOR.containing_type = _TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT
_TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT_SLOTVARIABLEREFERENCE.containing_type = _TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT
_TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT.fields_by_name['children'].message_type = _TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT_OBJECTREFERENCE
_TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT.fields_by_name['attributes'].message_type = _TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT_SERIALIZEDTENSOR
_TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT.fields_by_name['slot_variables'].message_type = _TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT_SLOTVARIABLEREFERENCE
_TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT.containing_type = _TRACKABLEOBJECTGRAPH
_TRACKABLEOBJECTGRAPH.fields_by_name['nodes'].message_type = _TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT
DESCRIPTOR.message_types_by_name['TrackableObjectGraph'] = _TRACKABLEOBJECTGRAPH
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TrackableObjectGraph = _reflection.GeneratedProtocolMessageType('TrackableObjectGraph', (_message.Message,), {

  'TrackableObject' : _reflection.GeneratedProtocolMessageType('TrackableObject', (_message.Message,), {

    'ObjectReference' : _reflection.GeneratedProtocolMessageType('ObjectReference', (_message.Message,), {
      'DESCRIPTOR' : _TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT_OBJECTREFERENCE,
      '__module__' : 'trackable_object_graph_pb2'
      # @@protoc_insertion_point(class_scope:tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference)
      })
    ,

    'SerializedTensor' : _reflection.GeneratedProtocolMessageType('SerializedTensor', (_message.Message,), {
      'DESCRIPTOR' : _TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT_SERIALIZEDTENSOR,
      '__module__' : 'trackable_object_graph_pb2'
      # @@protoc_insertion_point(class_scope:tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor)
      })
    ,

    'SlotVariableReference' : _reflection.GeneratedProtocolMessageType('SlotVariableReference', (_message.Message,), {
      'DESCRIPTOR' : _TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT_SLOTVARIABLEREFERENCE,
      '__module__' : 'trackable_object_graph_pb2'
      # @@protoc_insertion_point(class_scope:tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference)
      })
    ,
    'DESCRIPTOR' : _TRACKABLEOBJECTGRAPH_TRACKABLEOBJECT,
    '__module__' : 'trackable_object_graph_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.TrackableObjectGraph.TrackableObject)
    })
  ,
  'DESCRIPTOR' : _TRACKABLEOBJECTGRAPH,
  '__module__' : 'trackable_object_graph_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.TrackableObjectGraph)
  })
_sym_db.RegisterMessage(TrackableObjectGraph)
_sym_db.RegisterMessage(TrackableObjectGraph.TrackableObject)
_sym_db.RegisterMessage(TrackableObjectGraph.TrackableObject.ObjectReference)
_sym_db.RegisterMessage(TrackableObjectGraph.TrackableObject.SerializedTensor)
_sym_db.RegisterMessage(TrackableObjectGraph.TrackableObject.SlotVariableReference)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
