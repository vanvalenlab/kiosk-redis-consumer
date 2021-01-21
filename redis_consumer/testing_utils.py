# Copyright 2016-2020 The Van Valen Lab at the California Institute of
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
"""Common tests and fixtures for unit tests"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from skimage.external import tifffile as tiff

import pytest
import fakeredis

from redis_consumer import utils


@pytest.fixture
def redis_client():
    client = fakeredis.FakeStrictRedis(decode_responses='utf8')
    # patch the _redis_master field
    client._redis_master = client
    client._update_masters_and_slaves = lambda: True
    yield client


def _get_image(img_h=300, img_w=300, channels=None):
    shape = [img_w, img_h]
    if channels:
        shape.append(channels)
    shape = tuple(shape)
    bias = np.random.rand(*shape) * 64
    variance = np.random.rand(*shape) * (255 - 64)
    img = np.random.rand(*shape) * variance + bias
    return img


class Bunch(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


class DummyStorage(object):
    # pylint: disable=W0613,R0201
    def __init__(self, num=3):
        self.num = num

    def download(self, path, dest):
        if path.lower().endswith('.zip'):
            paths = []
            for i in range(self.num):
                img = _get_image()
                base, ext = os.path.splitext(path)
                _path = '{}{}{}'.format(base, i, ext)
                tiff.imsave(os.path.join(dest, _path), img)
                paths.append(_path)
            return utils.zip_files(paths, dest)
        img = _get_image()
        tiff.imsave(os.path.join(dest, path), img)
        return path

    def upload(self, zip_path, subdir=None):
        return 'zip_path.zip', 'blob.public_url'

    def get_public_url(self, zip_path):
        return 'blob.public_url'


def make_model_metadata_of_size(model_shape=(-1, 256, 256, 2)):

    def get_model_metadata(model_name, model_version):  # pylint: disable=unused-argument
        return [{
            'in_tensor_name': 'image',
            'in_tensor_dtype': 'DT_FLOAT',
            'in_tensor_shape': ','.join(str(s) for s in model_shape),
        }]
    return get_model_metadata
