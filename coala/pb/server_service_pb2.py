# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: coala/pb/server_service.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from coala.pb import common_pb2 as coala_dot_pb_dot_common__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1d\x63oala/pb/server_service.proto\x12\x08\x63oala.pb\x1a\x15\x63oala/pb/common.proto\"o\n\rUploadRequest\x12\x0f\n\x07task_id\x18\x01 \x01(\t\x12\x10\n\x08round_id\x18\x02 \x01(\x05\x12\x11\n\tclient_id\x18\x03 \x01(\t\x12(\n\x07\x63ontent\x18\x04 \x01(\x0b\x32\x17.coala.pb.UploadContent\"\x89\x01\n\rUploadContent\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12 \n\x04type\x18\x02 \x01(\x0e\x32\x12.coala.pb.DataType\x12\x11\n\tdata_size\x18\x03 \x01(\x03\x12&\n\x06metric\x18\x04 \x01(\x0b\x32\x16.coala.pb.ClientMetric\x12\r\n\x05\x65xtra\x18\x63 \x01(\x0c\"o\n\x0bPerformance\x12\x31\n\x06metric\x18\x01 \x03(\x0b\x32!.coala.pb.Performance.MetricEntry\x1a-\n\x0bMetricEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\"2\n\x0eUploadResponse\x12 \n\x06status\x18\x01 \x01(\x0b\x32\x10.coala.pb.Status\"V\n\nRunRequest\x12\r\n\x05model\x18\x01 \x01(\x0c\x12!\n\x07\x63lients\x18\x02 \x03(\x0b\x32\x10.coala.pb.Client\x12\x16\n\x0e\x65tcd_addresses\x18\x03 \x01(\t\"/\n\x0bRunResponse\x12 \n\x06status\x18\x01 \x01(\x0b\x32\x10.coala.pb.Status\"\r\n\x0bStopRequest\"0\n\x0cStopResponse\x12 \n\x06status\x18\x01 \x01(\x0b\x32\x10.coala.pb.Status\";\n\x06\x43lient\x12\x11\n\tclient_id\x18\x01 \x01(\t\x12\x0f\n\x07\x61\x64\x64ress\x18\x02 \x01(\t\x12\r\n\x05index\x18\x03 \x01(\x05\x32\xbd\x01\n\rServerService\x12=\n\x06Upload\x12\x17.coala.pb.UploadRequest\x1a\x18.coala.pb.UploadResponse\"\x00\x12\x34\n\x03Run\x12\x14.coala.pb.RunRequest\x1a\x15.coala.pb.RunResponse\"\x00\x12\x37\n\x04Stop\x12\x15.coala.pb.StopRequest\x1a\x16.coala.pb.StopResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'coala.pb.server_service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_PERFORMANCE_METRICENTRY']._loaded_options = None
  _globals['_PERFORMANCE_METRICENTRY']._serialized_options = b'8\001'
  _globals['_UPLOADREQUEST']._serialized_start=66
  _globals['_UPLOADREQUEST']._serialized_end=177
  _globals['_UPLOADCONTENT']._serialized_start=180
  _globals['_UPLOADCONTENT']._serialized_end=317
  _globals['_PERFORMANCE']._serialized_start=319
  _globals['_PERFORMANCE']._serialized_end=430
  _globals['_PERFORMANCE_METRICENTRY']._serialized_start=385
  _globals['_PERFORMANCE_METRICENTRY']._serialized_end=430
  _globals['_UPLOADRESPONSE']._serialized_start=432
  _globals['_UPLOADRESPONSE']._serialized_end=482
  _globals['_RUNREQUEST']._serialized_start=484
  _globals['_RUNREQUEST']._serialized_end=570
  _globals['_RUNRESPONSE']._serialized_start=572
  _globals['_RUNRESPONSE']._serialized_end=619
  _globals['_STOPREQUEST']._serialized_start=621
  _globals['_STOPREQUEST']._serialized_end=634
  _globals['_STOPRESPONSE']._serialized_start=636
  _globals['_STOPRESPONSE']._serialized_end=684
  _globals['_CLIENT']._serialized_start=686
  _globals['_CLIENT']._serialized_end=745
  _globals['_SERVERSERVICE']._serialized_start=748
  _globals['_SERVERSERVICE']._serialized_end=937
# @@protoc_insertion_point(module_scope)