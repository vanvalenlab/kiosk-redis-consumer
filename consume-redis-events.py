# Copyright 2016-2019 The Van Valen Lab at the California Institute of
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

import logging
import logging.handlers
import sys
import traceback

import redis_consumer
from redis_consumer import settings


def initialize_logger(debug_mode=True):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s]:[%(levelname)s]:[%(name)s]: %(message)s')
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)

    fh = logging.handlers.RotatingFileHandler(
        filename='redis-consumer.log',
        maxBytes=10000000,
        backupCount=10)
    fh.setFormatter(formatter)

    if debug_mode:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)
    fh.setLevel(logging.DEBUG)

    logger.addHandler(console)
    logger.addHandler(fh)


def get_consumer(consumer_type, **kwargs):
    logging.debug('Getting `%s` consumer with args %s.', consumer_type, kwargs)
    consumer_cls = redis_consumer.consumers.CONSUMERS.get(str(consumer_type).lower())
    if not consumer_cls:
        raise ValueError('Invalid `consumer_type`: "{}"'.format(consumer_type))
    return consumer_cls(**kwargs)


if __name__ == '__main__':
    initialize_logger(settings.DEBUG)

    _logger = logging.getLogger(__file__)

    redis = redis_consumer.redis.RedisClient(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        backoff=settings.REDIS_TIMEOUT)

    storage_client = redis_consumer.storage.get_client(settings.CLOUD_PROVIDER)

    consumer_kwargs = {
        'redis_client': redis,
        'storage_client': storage_client,
        'queue': settings.QUEUE,
        'final_status': 'done',
        'failed_status': 'failed',
        'name': settings.HOSTNAME,
        'output_dir': settings.OUTPUT_DIR,
    }

    _logger.debug('Getting `%s` consumer with args %s.',
                  settings.CONSUMER_TYPE, consumer_kwargs)

    consumer = get_consumer(settings.CONSUMER_TYPE, **consumer_kwargs)

    _logger.debug('Got `%s` consumer.', settings.CONSUMER_TYPE)

    while True:
        try:
            consumer.consume()
        except Exception as err:  # pylint: disable=broad-except
            _logger.critical('Fatal Error: %s: %s\n%s',
                             type(err).__name__, err, traceback.format_exc())

            sys.exit(1)
