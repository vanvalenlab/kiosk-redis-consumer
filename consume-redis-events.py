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
import time
import logging
import asyncio

import uvloop
from redis import StrictRedis

from redis_consumer import consumers
from redis_consumer import settings
from redis_consumer import storage
from redis_consumer.clients import TensorFlowServingClient
from redis_consumer.clients import DataProcessingClient


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
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    initialize_logger(settings.DEBUG)

    _logger = logging.getLogger(__file__)

    redis = StrictRedis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        decode_responses=True,
        charset='utf-8')

    storage_client = storage.get_client(settings.CLOUD_PROVIDER)
    dp_client = DataProcessingClient(settings.DP_HOST, settings.DP_PORT)
    tf_client = TensorFlowServingClient(settings.TF_HOST, settings.TF_PORT)

    consumer = consumers.PredictionConsumer(
        redis_client=redis,
        storage_client=storage_client,
        tf_client=tf_client,
        dp_client=dp_client,
        final_status='done')

    while True:
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(consumer.consume(
                settings.STATUS, settings.HASH_PREFIX))

            time.sleep(settings.INTERVAL)
        except Exception as err:  # pylint: disable=broad-except
            _logger.critical('Fatal Error: %s: %s', type(err).__name__, err)
            sys.exit(1)
