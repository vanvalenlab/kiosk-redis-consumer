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
TensorFlowServingClient POSTs data to the TensorFlow Serving Host
and returns the response. Raises TensorFlowServingError if request
is not fulfilled.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import logging

import numpy as np
import requests
import datetime

from tornado import httpclient
from tornado.gen import multi
from tornado import escape

class TensorFlowServingError(Exception):
    pass


class TensorFlowServingClient(object):
    """Class to interact with TensorFlow Serving"""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.logger = logging.getLogger(str(self.__class__.__name__))

    def get_url(self, model_name, version):
        """Get API URL for TensorFlow Serving, based on model name and version
        """
        return 'http://{}:{}/v1/models/{}/versions/{}:predict'.format(
            self.host, self.port, model_name, version)

    def fix_json(self, response_text):
        """Sometimes TF Serving has strange scientific notation e.g. '1e5.0,'
        so convert the float-exponent to an integer for JSON parsing.
        """
        self.logger.debug('tf-serving response is not well-formed JSON. '
                          'Attempting to fix the response.')
        try:
            fixed_text = response_text.replace('.0,', ',').replace('.0],', '],')
            fixed_json = json.loads(fixed_text)
            self.logger.debug('Successfully parsed tf-serving JSON response')
            return fixed_json
        except Exception as err:
            self.logger.error('Cannot fix tf-serving response JSON: %s', err)
            raise err

    async def tornado_images(self, images, model_name, model_version, timeout=300, max_clients=10):
        httpclient.AsyncHTTPClient.configure(None,
            max_body_size=1073741824,  # 1GB
            max_clients=max_clients)

        http_client = httpclient.AsyncHTTPClient()
        api_url = self.get_url(model_name, model_version)

        json_payload = ({ 'instances': [{ 'image': i.tolist() }] } for i in images)
        payloads = (escape.json_encode(jp) for jp in json_payload)

        def iter_kwargs(payload):
            for p in payload:
                yield {
                    'body': p,
                    'method': 'POST',
                    'request_timeout': timeout,
                    'connect_timeout': timeout
                }

        all_tf_results = []
        for kw in iter_kwargs(payloads):
            try:
                response = await http_client.fetch(api_url, **kw)
                text = response.body
                try:
                    prediction_json = json.loads(text)
                except:
                    prediction_json = self.fix_json(text)

                final_prediction = np.array(list(prediction_json['predictions'][0]))
                all_tf_results.append(final_prediction)
            except httpclient.HTTPError as err:
                self.logger.error('Error: %s: %s', err, err.response.body)
                raise err

        # # Using gen.multi - causes unpredictable 429 errors
        # requests = (http_client.fetch(api_url, **kw) for kw in iter_kwargs(payloads))
        # responses = await multi([r for r in requests])
        # texts = (escape.native_str(r.body) for r in responses)
        # for text in texts:
        #     try:
        #         prediction_json = json.loads(text)
        #     except:
        #         prediction_json = self.fix_json(text)
        #
        #     # Convert prediction to numpy array
        #     final_prediction = np.array(list(prediction_json['predictions'][0]))
        #     self.logger.debug('Got tf-serving results of shape: %s',
        #         final_prediction.shape)
        #     all_tf_results.append(final_prediction)
        return all_tf_results

    def post_image(self, image, model_name, version, timeout=300):
        """Sends image to tensorflow serving and returns response
        # Arguments:
            image: numpy array of image data passed to model
            model_name: hosted model to send image data
            version: model version to query
        # Returns: tf-serving results as numpy array
        """
        # Define payload to send to API URL
        payload = { 'instances': [{ 'image': image.tolist() }] }

        # Post to API URL
        return_code = 0
        retries = 0
        while return_code!=200:
            self.logger.debug('Sent request to tf-serving: %s',
                datetime.datetime.now())
            prediction = requests.post(
                self.get_url(model_name, version),
                json=payload,
                timeout=timeout)
            try:
                prediction_json = prediction.json()
            except:
                prediction_json = self.fix_json(prediction)
            # Check for tf-serving errors
            if not prediction.status_code == 200:
                if retries < 2:
                    retries += 1
                else:
                    self.logger.debug('Got response from tf-serving: %s',
                        datetime.datetime.now())
                    prediction_error = prediction.json()['error']

                    raise TensorFlowServingError('{}: {}'.format(
                        prediction_error, prediction.status_code))
            else:
                self.logger.debug('Got response from tf-serving: %s',
                    datetime.datetime.now())
                return_code = prediction.status_code

        # Convert prediction to numpy array
        final_prediction = np.array(list(prediction_json['predictions'][0]))
        self.logger.debug('Got tf-serving results of shape: %s',
            final_prediction.shape)

        return final_prediction
