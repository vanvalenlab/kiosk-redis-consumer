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
"""Tests for API Client classes"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import numpy as np
import pytest

from redis_consumer import clients


class TestClient(object):

    def test_get_url(self):
        client = clients.Client(host='localhost', port=80)
        with pytest.raises(NotImplementedError):
            _ = client.get_url()

    def test_format_image_payload(self):
        client = clients.Client(host='localhost', port=80)
        with pytest.raises(NotImplementedError):
            img = np.zeros((30, 30, 1))
            _ = client.format_image_payload(img)

    def test_handle_response(self):
        client = clients.Client(host='localhost', port=80)
        with pytest.raises(NotImplementedError):
            _ = client.handle_response({})


class TestTensorFlowServingClient(object):

    def _get_client(self):
        return clients.TensorFlowServingClient(host='localhost', port=8501)

    def test_get_url(self):
        name = 'model_name'
        version = '1'
        client = self._get_client()

        url_should_be = 'http://{}:{}/v1/models/{}/versions/{}:predict'.format(
            client.host, client.port, name, version)
        assert client.get_url(name, version) == url_should_be

    def test_format_image_payload(self):
        client = self._get_client()
        img = np.zeros((30, 30, 1))
        payload = client.format_image_payload(img)
        assert 'instances' in payload
        instances = payload['instances']
        assert isinstance(instances, (list,))
        assert len(instances) == 1
        assert 'image' in instances[0]
        assert len(instances[0]['image']) == 30
        assert len(instances[0]['image'][0]) == 30
        assert len(instances[0]['image'][0][1]) == 1

        with pytest.raises(Exception):
            client.format_image_payload(None)

    def test_handle_response(self):
        shape = (30, 30, 1)
        response = {'predictions': [np.random.random(shape).tolist()]}
        client = self._get_client()
        handled = client.handle_response(response)
        assert isinstance(handled, (np.ndarray, np.generic))
        assert handled.shape == (30, 30, 1)

    def test_fix_json(self):
        jsonstring = '{"image": [1e1.0, 4e1]}'
        clean_json = self._get_client().fix_json(jsonstring)
        assert isinstance(clean_json, dict)

        with pytest.raises(Exception):
            badjson = '{"notvalid": "Json"}, [1,2,3]'
            _ = self._get_client().fix_json(badjson)
