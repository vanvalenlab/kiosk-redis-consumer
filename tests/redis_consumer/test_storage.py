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
"""Tests for API Storage classes"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import pytest

from redis_consumer import storage
from redis_consumer import utils


# def test_get_client():
#     aws = storage.get_client('aws')
#     AWS = storage.get_client('AWS')
#     assert isinstance(aws, type(AWS))

#     # TODO: set GCLOUD env vars to test this
#     with pytest.raises(OSError):
#         gke = storage.get_client('gke')
#         GKE = storage.get_client('GKE')
#         assert isinstance(gke, type(GKE))

#     with pytest.raises(ValueError):
#         _ = storage.get_client('bad_value')


class TestStorage(object):

    def test_get_download_path(self):
        with utils.get_tempdir() as tempdir:
            bucket = 'test-bucket'
            stg = storage.Storage(bucket, tempdir)
            filekey = 'upload_dir/key/to.zip'
            path = stg.get_download_path(filekey, tempdir)
            path2 = stg.get_download_path(filekey)
            assert path == path2
            assert str(path).startswith(tempdir)
            assert str(path).endswith(filekey.replace('upload_dir/', ''))
