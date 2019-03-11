# Copyright 2016-2018 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
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
# distributed under the License is distributed on an "AS IS" BASIS,
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


# remove leading/trailing "/"s from cloud bucket folder names
_strip = lambda x: '/'.join(y for y in x.split('/') if y)

# Debug Mode
DEBUG = config('DEBUG', cast=bool, default=False)

# Consumer settings
INTERVAL = config('INTERVAL', default=10, cast=int)
CONSUMER_TYPE = config('CONSUMER_TYPE', default='image')

# Hash Prefix - filter out prediction jobs
HASH_PREFIX = _strip(config('HASH_PREFIX', cast=str, default='predict'))

# Redis client connection
REDIS_HOST = config('REDIS_HOST', default='redis-master')
REDIS_PORT = config('REDIS_PORT', default=6379, cast=int)

# tensorflow-serving client connection
TF_HOST = config('TF_HOST', default='tf-serving-service')
TF_PORT = config('TF_PORT', default=8500, cast=int)
TF_TENSOR_NAME = config('TF_TENSOR_NAME', default='image')
TF_TENSOR_DTYPE = config('TF_TENSOR_DTYPE', default='DT_FLOAT')

# data-processing client connection
DP_HOST = config('DP_HOST', default='data-processing-service')
DP_PORT = config('DP_PORT', default=8080, cast=int)

# gRPC API timeout in seconds (scales with `cuts`)
GRPC_TIMEOUT = config('GRPC_TIMEOUT', default=30, cast=int)
REDIS_TIMEOUT = config('REDIS_TIMEOUT', default=3, cast=int)

# Status of hashes marked for prediction
STATUS = config('STATUS', default='new')

# Cloud storage
CLOUD_PROVIDER = config('CLOUD_PROVIDER', cast=str, default='aws').lower()

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
HOSTNAME = config('HOSTMANE', default="host-unkonwn")
