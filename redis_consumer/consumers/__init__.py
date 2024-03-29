# Copyright 2016-2022 The Van Valen Lab at the California Institute of
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Base class consumers
from redis_consumer.consumers.base_consumer import Consumer
from redis_consumer.consumers.base_consumer import TensorFlowServingConsumer
from redis_consumer.consumers.base_consumer import ZipFileConsumer

# Custom Workflow consumers
from redis_consumer.consumers.segmentation_consumer import SegmentationConsumer
from redis_consumer.consumers.caliban_consumer import CalibanConsumer
from redis_consumer.consumers.mesmer_consumer import MesmerConsumer
from redis_consumer.consumers.polaris_consumer import PolarisConsumer
from redis_consumer.consumers.spot_consumer import SpotConsumer
# TODO: Import future custom Consumer classes.


CONSUMERS = {
    'image': SegmentationConsumer,  # deprecated, use "segmentation" instead.
    'segmentation': SegmentationConsumer,
    'zip': ZipFileConsumer,
    'tracking': CalibanConsumer,  # deprecated, use "caliban" instead.
    'multiplex': MesmerConsumer,  # deprecated, use "mesmer" instead.
    'mesmer': MesmerConsumer,
    'caliban': CalibanConsumer,
    'polaris': PolarisConsumer,
    'spot': SpotConsumer,
    # TODO: Add future custom Consumer classes here.
}


# Backwards compatibility aliases
MultiplexConsumer = MesmerConsumer
TrackingConsumer = CalibanConsumer


del absolute_import
del division
del print_function
