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

import json
import logging
import time

import numpy as np
import requests
import datetime

from tornado import httpclient
from tornado import escape
from tornado.gen import multi


class TensorFlowServingError(Exception):
    """Custom error for TensorFlowServing"""
    pass


class TensorFlowServingClient(object):
    """Class to interact with TensorFlow Serving"""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.logger = logging.getLogger(str(self.__class__.__name__))

    def verify_endpoint_liveness(self,
                                 num_retries=60,
                                 timeout=10,
                                 retry_interval=10):
        """Ping TensorFlow Serving to check if the service is up.
        # Arguments:
            num_retries: number of attempts to ping tf serving
            timeout: timeout in seconds for each ping attempt
            retry_interval: time to wait between retries
        # Returns:
            True if tf serving service is live otherwise False
        """
        liveness_url = 'http://{}:{}'.format(self.host, self.port)

        for i in range(num_retries):
            try:
                response = requests.get(liveness_url, timeout=timeout)
                if response.status_code == 404:
                    self.logger.debug('Connection to tf-serving established '
                                      ' after %s attempts.', i + 1)
                    break

                self.logger.error('Expected a 404 response but got %s. '
                                  'Entered `unreachable` code block.',
                                  response.status_code)

            except Exception as err:
                self.logger.warning('Encountered %s while checking tf-serving '
                                    ' liveness:  %s', type(err).__name__, err)

            # sleep as long as needed to allow tf-serving time to startup
            time.sleep(retry_interval)

        else:  # for/else loop.  only enters block after all retries
            self.logger.error('Connection to tf-serving not established. '
                              'Exhausted all %s retries.', num_retries)
            return False
        return True

    def get_url(self, model_name, model_version):
        """Get API URL for TensorFlow Serving, based on model name and version
        # Arguments:
            model_name: name of model hosted by tf-serving
            model_version: integer version of `model_name`
        # Returns:
            formatted URL for HTTP request
        """
        return 'http://{}:{}/v1/models/{}/versions/{}:{}'.format(
            self.host, self.port, model_name, model_version, 'predict')

    def fix_json(self, response_text):
        """Sometimes TF Serving has strange scientific notation e.g. '1e5.0,'
        so convert the float-exponent to an integer for JSON parsing.
        # Arguments:
            response_text: HTTP response as string
        # Returns:
            fixed_json: HTTP response as JSON object
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

    async def tornado_images(self,
                             images,
                             model_name,
                             model_version,
                             timeout=300,
                             max_clients=10):
        """POSTs many images to tf-serving at once using tornado.
        # Arguments:
            images: list of image data to pass to tf-serving
            model_name: hosted model to send image data
            model_version: model version to query
            timeout: total timeout for all requests
            max_clients: max number of simultaneous http clients
        # Returns:
            all_tf_results: list of results from tf-serving
        """
        httpclient.AsyncHTTPClient.configure(
            None,
            max_body_size=1073741824,  # 1GB
            max_clients=max_clients)

        http_client = httpclient.AsyncHTTPClient()
        api_url = self.get_url(model_name, model_version)

        json_payload = ({'instances': [{'image': i.tolist()}]} for i in images)
        payloads = (escape.json_encode(jp) for jp in json_payload)

        def iter_kwargs(payload):
            for pyld in payload:
                yield {
                    'body': pyld,
                    'method': 'POST',
                    'request_timeout': timeout,
                    'connect_timeout': timeout
                }

        all_tf_results = []
        for kwargs in iter_kwargs(payloads):
            try:
                response = await http_client.fetch(api_url, **kwargs)
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

    def post_image(self,
                   image,
                   model_name,
                   model_version,
                   timeout=300,
                   num_retries=3):
        """Sends image to tensorflow serving and returns response
        # Arguments:
            image: numpy array of image data passed to model
            model_name: hosted model to send image data
            model_version: model version to query
        # Returns:
            tf-serving results as numpy array
        """
        # Define payload to send to API URL
        payload = {'instances': [{'image': image.tolist()}]}

        # Post to API URL
        for i in range(num_retries):
            self.logger.debug('Sending request to tf-serving: %s',
                datetime.datetime.now())

            prediction = requests.post(
                self.get_url(model_name, model_version),
                json=payload,
                timeout=timeout)

            self.logger.debug('Got response from tf-serving: %s',
                datetime.datetime.now())

            try:
                prediction_json = prediction.json()
            except:
                prediction_json = self.fix_json(prediction)

            # Check for tf-serving errors
            if prediction.status_code == 200:
                break  # prediction is found, exit the loop
            else:  # tf-serving error.  Retry or raise it.
                if i < num_retries - 1:
                    self.logger.warning('TensorFlow Serving request %s failed'
                                        ' due to error %s. Retrying...',
                                        prediction_json['error'], i)
                else:
                    raise TensorFlowServingError('{}: {}'.format(
                        prediction_json['error'], prediction.status_code))

        # Convert prediction to numpy array
        final_prediction = np.array(list(prediction_json['predictions'][0]))
        self.logger.debug('Got tf-serving results of shape: %s',
                          final_prediction.shape)

        return final_prediction
