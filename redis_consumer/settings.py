# Copyright 2016-2021 The Van Valen Lab at the California Institute of
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

from decouple import config

import deepcell


# Application directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR = os.path.join(ROOT_DIR, 'download')

# Consumer settings
INTERVAL = config('INTERVAL', default=10, cast=int)
CONSUMER_TYPE = config('CONSUMER_TYPE', default='image')
MAX_RETRY = config('MAX_RETRY', default=5, cast=int)
MAX_IMAGE_HEIGHT = config('MAX_IMAGE_HEIGHT', default=2048, cast=int)
MAX_IMAGE_WIDTH = config('MAX_IMAGE_WIDTH', default=2048, cast=int)
MAX_IMAGE_FRAMES = config('MAX_IMAGE_FRAMES', default=60, cast=int)

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

# timeout/backoff wait time in seconds
REDIS_TIMEOUT = config('REDIS_TIMEOUT', default=3, cast=int)
EMPTY_QUEUE_TIMEOUT = config('EMPTY_QUEUE_TIMEOUT', default=5, cast=int)
DO_NOTHING_TIMEOUT = config('DO_NOTHING_TIMEOUT', default=0.5, cast=float)
STORAGE_MAX_BACKOFF = config('STORAGE_MAX_BACKOFF', default=60, cast=float)

# AWS Credentials
AWS_REGION = config('AWS_REGION', default='us-east-1')
AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID', default='specify_me')
AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY', default='specify_me')

# Cloud Storage Bucket
STORAGE_BUCKET = config('STORAGE_BUCKET', default='s3://default-bucket')

# Pod Meteadta
HOSTNAME = config('HOSTNAME', default='host-unkonwn')

# Redis queue
QUEUE = config('QUEUE', default='predict')
SEGMENTATION_QUEUE = config('SEGMENTATION_QUEUE', default='predict')

# Configure expiration time for child keys
EXPIRE_TIME = config('EXPIRE_TIME', default=3600, cast=int)

# Configure expiration for cached model metadata
METADATA_EXPIRE_TIME = config('METADATA_EXPIRE_TIME', default=30, cast=int)

# Tracking settings
TRACKING_MODEL = config('TRACKING_MODEL', default='TrackingModelInf:4', cast=str)
CALIBAN_MODEL = config('CALIBAN_MODEL', default=TRACKING_MODEL, cast=str)
NEIGHBORHOOD_ENCODER = config('NEIGHBORHOOD_ENCODER', default='TrackingModelNE:2', cast=str)

# tracking.cell_tracker settings TODO: can we extract from model_metadata?
MAX_DISTANCE = config('MAX_DISTANCE', default=50, cast=int)
TRACK_LENGTH = config('TRACK_LENGTH', default=8, cast=int)
DIVISION = config('DIVISION', default=0.9, cast=float)
BIRTH = config('BIRTH', default=0.99, cast=float)
DEATH = config('DEATH', default=0.99, cast=float)

# Scale detection settings
SCALE_DETECT_MODEL = config('SCALE_DETECT_MODEL', default='ScaleDetection:1')
SCALE_DETECT_ENABLED = config('SCALE_DETECT_ENABLED', default=False, cast=bool)
MAX_SCALE = config('MAX_SCALE', default=3, cast=float)
MIN_SCALE = config('MIN_SCALE', default=1 / MAX_SCALE, cast=float)

# Type detection settings
LABEL_DETECT_MODEL = config('LABEL_DETECT_MODEL', default='LabelDetection:1', cast=str)
LABEL_DETECT_ENABLED = config('LABEL_DETECT_ENABLED', default=False, cast=bool)

# Mesmer model Settings
# deprecated model name, use MESMER_MODEL instead.
MULTIPLEX_MODEL = config('MULTIPLEX_MODEL', default='MultiplexSegmentation:5', cast=str)
MESMER_MODEL = config('MESMER_MODEL', default=MULTIPLEX_MODEL, cast=str)
MESMER_COMPARTMENT = config('MESMER_COMPARTMENT', default='whole-cell')

# Polaris model Settings
POLARIS_MODEL = config('POLARIS_MODEL', default='SpotDetection:3', cast=str)
POLARIS_THRESHOLD = config('POLARIS_THRESHOLD', default=0.95, cast=float)
POLARIS_CLIP = config('POLARIS_CLIP', default=False, cast=bool)

# Set default models based on label type
MODEL_CHOICES = {
    0: config('NUCLEAR_MODEL', default='NuclearSegmentation:0', cast=str),
    1: config('PHASE_MODEL', default='PhaseCytoSegmentation:0', cast=str),
    2: config('CYTOPLASM_MODEL', default='FluoCytoSegmentation:0', cast=str)
}

APPLICATION_CHOICES = {
    0: deepcell.applications.NuclearSegmentation,
    1: deepcell.applications.CytoplasmSegmentation,
    2: deepcell.applications.CytoplasmSegmentation
}
