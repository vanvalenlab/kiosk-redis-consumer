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
"""
Consume all unprocessed events in the redis queue and update each
with results from tensorflow-serving.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import logging

from redis import StrictRedis

from redis_consumer import consumers
from redis_consumer import settings
from redis_consumer import storage
from redis_consumer.tf_client import TensorFlowServingClient

import requests

def consume_predictions():
    redis = StrictRedis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        decode_responses=True,
        charset='utf-8')
    
    if settings.CLOUD_PROVIDER == 'aws':
        storage_client = storage.S3Storage(settings.AWS_S3_BUCKET)
    elif settings.CLOUD_PROVIDER == 'gke':
        storage_client = storage.GoogleStorage(settings.GCLOUD_STORAGE_BUCKET)
    else:
        print('Bad value for CLOUD_PROVIDER:', settings.CLOUD_PROVIDER)
        storage_client = None

    consumer = consumers.PredictionConsumer(
        redis_client=redis,
        storage_client=storage_client,
        tf_client=TensorFlowServingClient(settings.TF_HOST, settings.TF_PORT))

    consumer.consume(interval=10)


def initialize_logger(debug_mode=False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(levelname)s]:[%(name)s]: %(message)s')
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)

    if debug_mode:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)

    logger.addHandler(console)


if __name__ == '__main__':
    initialize_logger(settings.DEBUG)

    try:
        consume_predictions()
    except Exception as err:
        print(err)
        sys.exit(1)
