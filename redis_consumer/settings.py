# Copyright 2016-2020 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the 'License');
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
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Settings file to hold environment variabls and constants"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import grpc
from decouple import config

from redis_consumer import processing


# remove leading/trailing '/'s from cloud bucket folder names
def _strip(x):
    return '/'.join(y for y in x.split('/') if y)


# Debug Mode
DEBUG = config('DEBUG', cast=bool, default=False)

# Consumer settings
INTERVAL = config('INTERVAL', default=10, cast=int)
CONSUMER_TYPE = config('CONSUMER_TYPE', default='image')
MAX_RETRY = config('MAX_RETRY', default=5, cast=int)

# Redis client connection
REDIS_HOST = config('REDIS_HOST', default='redis-master')
REDIS_PORT = config('REDIS_PORT', default=6379, cast=int)

# tensorflow-serving client connection
TF_HOST = config('TF_HOST', default='tf-serving')
TF_PORT = config('TF_PORT', default=8500, cast=int)
TF_TENSOR_NAME = config('TF_TENSOR_NAME', default='image')
TF_TENSOR_DTYPE = config('TF_TENSOR_DTYPE', default='DT_FLOAT')

# data-processing client connection
DP_HOST = config('DP_HOST', default='data-processing')
DP_PORT = config('DP_PORT', default=8080, cast=int)

# gRPC API timeout in seconds (scales with `cuts`)
GRPC_TIMEOUT = config('GRPC_TIMEOUT', default=30, cast=int)
GRPC_BACKOFF = config('GRPC_BACKOFF', default=3, cast=int)

# Retry-able gRPC status codes
GRPC_RETRY_STATUSES = {
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.UNAVAILABLE
}

# timeout/backoff wait time in seconds
REDIS_TIMEOUT = config('REDIS_TIMEOUT', default=3, cast=int)
EMPTY_QUEUE_TIMEOUT = config('EMPTY_QUEUE_TIMEOUT', default=5, cast=int)

# Cloud storage
CLOUD_PROVIDER = config('CLOUD_PROVIDER', cast=str, default='gke').lower()

# Application directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR = os.path.join(ROOT_DIR, 'download')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')

for d in (DOWNLOAD_DIR, OUTPUT_DIR, LOG_DIR):
    try:
        os.mkdir(d)
    except OSError:
        pass

# AWS Credentials
AWS_REGION = config('AWS_REGION', default='us-east-1')
AWS_S3_BUCKET = config('AWS_S3_BUCKET', default='default-bucket')
AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID', default='specify_me')
AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY', default='specify_me')

# Google Credentials
GCLOUD_STORAGE_BUCKET = config('GKE_BUCKET', default='default-bucket')

# Pod Meteadta
HOSTNAME = config('HOSTNAME', default='host-unkonwn')

# Redis queue
QUEUE = config('QUEUE', default='predict')
SEGMENTATION_QUEUE = config('SEGMENTATION_QUEUE', default='predict')

# Configure expiration time for child keys
EXPIRE_TIME = config('EXPIRE_TIME', default=3600, cast=int)

# Pre- and Post-processing settings
PROCESSING_FUNCTIONS = {
    'pre': {
        'normalize': processing.normalize
    },
    'post': {
        'deepcell': processing.pixelwise,  # TODO: this is deprecated.
        'pixelwise': processing.pixelwise,
        'mibi': processing.mibi,
        'watershed': processing.watershed,
        'retinanet': processing.retinanet_to_label_image,
        'retinanet-semantic': processing.retinanet_semantic_to_label_image,
        'deep_watershed': processing.deep_watershed,
    },
}

# Tracking settings
TRACKING_SEGMENT_MODEL = config('TRACKING_SEGMENT_MODEL', default='panoptic:3', cast=str)
TRACKING_POSTPROCESS_FUNCTION = config('TRACKING_POSTPROCESS_FUNCTION',
                                       default='retinanet', cast=str)
CUTS = config('CUTS', default=0, cast=int)

TRACKING_MODEL = config('TRACKING_MODEL', default='TrackingModel:0', cast=str)

DRIFT_CORRECT_ENABLED = config('DRIFT_CORRECT_ENABLED', default=False, cast=bool)
NORMALIZE_TRACKING = config('NORMALIZE_TRACKING', default=True, cast=bool)

# tracking.cell_tracker settings
MAX_DISTANCE = config('MAX_DISTANCE', default=50, cast=int)
TRACK_LENGTH = config('TRACK_LENGTH', default=5, cast=int)
DIVISION = config('DIVISION', default=0.9, cast=float)
BIRTH = config('BIRTH', default=0.95, cast=float)
DEATH = config('DEATH', default=0.95, cast=float)
NEIGHBORHOOD_SCALE_SIZE = config('NEIGHBORHOOD_SCALE_SIZE', default=30, cast=int)

# Scale detection settings
SCALE_DETECT_MODEL = config('SCALE_DETECT_MODEL', default='ScaleDetection:3')
SCALE_DETECT_SAMPLE = config('SCALE_DETECT_SAMPLE', default=3, cast=int)
# Not supported for tracking. Always detects scale
SCALE_DETECT_ENABLED = config('SCALE_DETECT_ENABLED', default=False, cast=bool)
SCALE_RESHAPE_SIZE = config('SCALE_RESHAPE_SIZE', default=216, cast=int)

# Type detection settings
LABEL_DETECT_MODEL = config('LABEL_DETECT_MODEL', default='LabelDetection:2', cast=str)
LABEL_DETECT_SAMPLE = config('LABEL_DETECT_SAMPLE', default=3, cast=int)
LABEL_DETECT_ENABLED = config('LABEL_DETECT_ENABLED', default=False, cast=bool)
LABEL_RESHAPE_SIZE = config('LABEL_RESHAPE_SIZE', default=216, cast=int)

# Set default models based on label type
PHASE_MODEL = config('PHASE_MODEL', default='panoptic_phase:0', cast=str)
CYTOPLASM_MODEL = config('CYTOPLASM_MODEL', default='panoptic_cytoplasm:0', cast=str)
NUCLEAR_MODEL = config('NUCLEAR_MODEL', default='panoptic:3', cast=str)

MODEL_CHOICES = {
    0: NUCLEAR_MODEL,
    1: PHASE_MODEL,
    2: CYTOPLASM_MODEL
}

PHASE_POSTPROCESS = config('PHASE_POSTPROCESS', default='deep_watershed', cast=str)
CYTOPLASM_POSTPROCESS = config('CYTOPLASM_POSTPROCESS', default='deep_watershed', cast=str)
NUCLEAR_POSTPROCESS = config('NUCLEAR_POSTPROCESS', default='deep_watershed', cast=str)

PHASE_RESHAPE_SIZE = config('PHASE_RESHAPE_SIZE', default=512, cast=int)
CYTOPLASM_RESHAPE_SIZE = config('CYTOPLASM_RESHAPE_SIZE', default=512, cast=int)
NUCLEAR_RESHAPE_SIZE = config('NUCLEAR_RESHAPE_SIZE', default=512, cast=int)

MODEL_SIZES = {
    NUCLEAR_MODEL: NUCLEAR_RESHAPE_SIZE,
    PHASE_MODEL: PHASE_RESHAPE_SIZE,
    CYTOPLASM_MODEL: CYTOPLASM_RESHAPE_SIZE,
}

POSTPROCESS_CHOICES = {
    0: NUCLEAR_POSTPROCESS,
    1: PHASE_POSTPROCESS,
    2: CYTOPLASM_POSTPROCESS
}
