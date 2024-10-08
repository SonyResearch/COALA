# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from coala.pb import server_service_pb2 as coala_dot_pb_dot_server__service__pb2

GRPC_GENERATED_VERSION = '1.64.1'
GRPC_VERSION = grpc.__version__
EXPECTED_ERROR_RELEASE = '1.65.0'
SCHEDULED_RELEASE_DATE = 'June 25, 2024'
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    warnings.warn(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in coala/pb/server_service_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
        + f' This warning will become an error in {EXPECTED_ERROR_RELEASE},'
        + f' scheduled for release on {SCHEDULED_RELEASE_DATE}.',
        RuntimeWarning
    )


class ServerServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Upload = channel.unary_unary(
                '/coala.pb.ServerService/Upload',
                request_serializer=coala_dot_pb_dot_server__service__pb2.UploadRequest.SerializeToString,
                response_deserializer=coala_dot_pb_dot_server__service__pb2.UploadResponse.FromString,
                _registered_method=True)
        self.Run = channel.unary_unary(
                '/coala.pb.ServerService/Run',
                request_serializer=coala_dot_pb_dot_server__service__pb2.RunRequest.SerializeToString,
                response_deserializer=coala_dot_pb_dot_server__service__pb2.RunResponse.FromString,
                _registered_method=True)
        self.Stop = channel.unary_unary(
                '/coala.pb.ServerService/Stop',
                request_serializer=coala_dot_pb_dot_server__service__pb2.StopRequest.SerializeToString,
                response_deserializer=coala_dot_pb_dot_server__service__pb2.StopResponse.FromString,
                _registered_method=True)


class ServerServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Upload(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Run(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Stop(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ServerServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Upload': grpc.unary_unary_rpc_method_handler(
                    servicer.Upload,
                    request_deserializer=coala_dot_pb_dot_server__service__pb2.UploadRequest.FromString,
                    response_serializer=coala_dot_pb_dot_server__service__pb2.UploadResponse.SerializeToString,
            ),
            'Run': grpc.unary_unary_rpc_method_handler(
                    servicer.Run,
                    request_deserializer=coala_dot_pb_dot_server__service__pb2.RunRequest.FromString,
                    response_serializer=coala_dot_pb_dot_server__service__pb2.RunResponse.SerializeToString,
            ),
            'Stop': grpc.unary_unary_rpc_method_handler(
                    servicer.Stop,
                    request_deserializer=coala_dot_pb_dot_server__service__pb2.StopRequest.FromString,
                    response_serializer=coala_dot_pb_dot_server__service__pb2.StopResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'coala.pb.ServerService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('coala.pb.ServerService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class ServerService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Upload(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/coala.pb.ServerService/Upload',
            coala_dot_pb_dot_server__service__pb2.UploadRequest.SerializeToString,
            coala_dot_pb_dot_server__service__pb2.UploadResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Run(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/coala.pb.ServerService/Run',
            coala_dot_pb_dot_server__service__pb2.RunRequest.SerializeToString,
            coala_dot_pb_dot_server__service__pb2.RunResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Stop(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/coala.pb.ServerService/Stop',
            coala_dot_pb_dot_server__service__pb2.StopRequest.SerializeToString,
            coala_dot_pb_dot_server__service__pb2.StopResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
