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
MAX_IMAGE_HEIGHT = config('MAX_IMAGE_HEIGHT', default=2048, cast=int)
MAX_IMAGE_WIDTH = config('MAX_IMAGE_WIDTH', default=2048, cast=int)

# Redis client connection
REDIS_HOST = config('REDIS_HOST', default='redis-master')
REDIS_PORT = config('REDIS_PORT', default=6379, cast=int)

# TensorFlow Serving client connection
TF_HOST = config('TF_HOST', default='tf-serving')
TF_PORT = config('TF_PORT', default=8500, cast=int)
# maximum batch allowed by TensorFlow Serving
TF_MAX_BATCH_SIZE = config('TF_MAX_BATCH_SIZE', default=128, cast=int)
# minimum expected model size, dynamically change batches proportionately.
TF_MIN_MODEL_SIZE = config('TF_MIN_MODEL_SIZE', default=128, cast=int)

# gRPC API timeout in seconds
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
DO_NOTHING_TIMEOUT = config('DO_NOTHING_TIMEOUT', default=0.5, cast=float)
STORAGE_MAX_BACKOFF = config('STORAGE_MAX_BACKOFF', default=60, cast=float)

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

# Configure expiration for cached model metadata
METADATA_EXPIRE_TIME = config('METADATA_EXPIRE_TIME', default=30, cast=int)

# Pre- and Post-processing settings
PROCESSING_FUNCTIONS = {
    'pre': {
        'normalize': processing.normalize,
        'histogram_normalization': processing.phase_preprocess,
        'multiplex_preprocess': processing.multiplex_preprocess,
        'none': lambda x: x
    },
    'post': {
        'deepcell': processing.pixelwise,  # TODO: this is deprecated.
        'pixelwise': processing.pixelwise,
        'watershed': processing.watershed,
        'retinanet': processing.retinanet_to_label_image,
        'retinanet-semantic': processing.retinanet_semantic_to_label_image,
        'deep_watershed': processing.deep_watershed,
        'multiplex_postprocess_consumer': processing.multiplex_postprocess_consumer,
        'none': lambda x: x
    },
}

# Tracking settings
TRACKING_SEGMENT_MODEL = config('TRACKING_SEGMENT_MODEL', default='panoptic:3', cast=str)
TRACKING_POSTPROCESS_FUNCTION = config('TRACKING_POSTPROCESS_FUNCTION',
                                       default='retinanet', cast=str)

TRACKING_MODEL = config('TRACKING_MODEL', default='TrackingModel:0', cast=str)

DRIFT_CORRECT_ENABLED = config('DRIFT_CORRECT_ENABLED', default=False, cast=bool)
NORMALIZE_TRACKING = config('NORMALIZE_TRACKING', default=True, cast=bool)

# tracking.cell_tracker settings TODO: can we extract from model_metadata?
MAX_DISTANCE = config('MAX_DISTANCE', default=50, cast=int)
TRACK_LENGTH = config('TRACK_LENGTH', default=5, cast=int)
DIVISION = config('DIVISION', default=0.9, cast=float)
BIRTH = config('BIRTH', default=0.95, cast=float)
DEATH = config('DEATH', default=0.95, cast=float)
NEIGHBORHOOD_SCALE_SIZE = config('NEIGHBORHOOD_SCALE_SIZE', default=30, cast=int)

MAX_SCALE = config('MAX_SCALE', default=3, cast=float)
MIN_SCALE = config('MIN_SCALE', default=1 / MAX_SCALE, cast=float)

# Scale detection settings
SCALE_DETECT_MODEL = config('SCALE_DETECT_MODEL', default='ScaleDetection:1')
SCALE_DETECT_ENABLED = config('SCALE_DETECT_ENABLED', default=False, cast=bool)

# Type detection settings
LABEL_DETECT_MODEL = config('LABEL_DETECT_MODEL', default='LabelDetection:1', cast=str)
LABEL_DETECT_ENABLED = config('LABEL_DETECT_ENABLED', default=False, cast=bool)

# Multiplex model Settings
MULTIPLEX_MODEL = config('MULTIPLEX_MODEL', default='MultiplexSegmentation:5', cast=str)

# Set default models based on label type
MODEL_CHOICES = {
    0: config('NUCLEAR_MODEL', default='NuclearSegmentation:0', cast=str),
    1: config('PHASE_MODEL', default='PhaseCytoSegmentation:0', cast=str),
    2: config('CYTOPLASM_MODEL', default='FluoCytoSegmentation:0', cast=str)
}

PREPROCESS_CHOICES = {
    0: config('NUCLEAR_PREPROCESS', default='normalize', cast=str),
    1: config('PHASE_PREPROCESS', default='histogram_normalization', cast=str),
    2: config('CYTOPLASM_PREPROCESS', default='histogram_normalization', cast=str)
}

POSTPROCESS_CHOICES = {
    0: config('NUCLEAR_POSTPROCESS', default='deep_watershed', cast=str),
    1: config('PHASE_POSTPROCESS', default='deep_watershed', cast=str),
    2: config('CYTOPLASM_POSTPROCESS', default='deep_watershed', cast=str)
}
