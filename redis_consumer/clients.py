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
"""Client classes for various APIs used by the consumer
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


# Custom Exceptions

class TensorFlowServingError(Exception):
    """Custom error for TensorFlowServing"""
    pass


class DataProcessingError(Exception):
    """Custom error for DataProcessing API"""
    pass


# Client classes

class Client(object):
    """Abstract Base class for API Clients"""

    def __init__(self,
                 host,
                 port,
                 max_body_size=1073741824,  # 1GB
                 health_route=None):
        self.host = host
        self.port = port
        self.health_route = health_route
        self.max_body_size = max_body_size
        self.logger = logging.getLogger(str(self.__class__.__name__))

    def get_url(self, *args):
        """Based on the inputs, return a formatted API URL"""
        raise NotImplementedError

    def verify_endpoint_liveness(self,
                                 num_retries=60,
                                 timeout=10,
                                 retry_interval=10,
                                 expected_code=200):
        """Ping API to check if the service is up.
        # Arguments:
            num_retries: number of attempts to ping
            timeout: timeout in seconds for each ping attempt
            retry_interval: time to wait between retries
        # Returns:
            True if service is live otherwise False
        """
        liveness_url = 'http://{}:{}'.format(self.host, self.port)
        if self.health_route is not None:
            liveness_url = '{}/{}'.format(liveness_url, self.health_route)

        for i in range(num_retries):
            try:
                response = requests.get(liveness_url, timeout=timeout)
                if response.status_code == expected_code:
                    self.logger.debug('Connection to tf-serving established '
                                      ' after %s attempts.', i + 1)
                    break

                self.logger.error('Expected a %s response but got %s. '
                                  'Entered `unreachable` code block.',
                                  expected_code, response.status_code)

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

    def format_image_payload(self, image):
        """Format the payload of the list of numpy image data.
        # TODO: Placeholder - implement for each subclass
        # Arguments:
            images: list of np arrays of image data
        # Returns:
            Formatted payload for the specific API
        """
        raise NotImplementedError

    def handle_tornado_response(self, response):
        """Handle the API response.
        Each client will have a different implementation
        # Arguments:
            response: response from the tornado http client
        # Returns:
            result: data parsed from the API repsonse
        """
        raise NotImplementedError

    async def tornado_images(self, images, url, timeout=300, max_clients=10):
        """POSTs many images to the API at once using tornado.
        # Arguments:
            images: list of image data to pass to the API
            url: URL to send the request
            timeout: total timeout for all requests
            max_clients: max number of simultaneous http clients
        # Returns:
            results: list of results from API
        """
        # Create the HTTP Client
        httpclient.AsyncHTTPClient.configure(
            None,
            max_body_size=self.max_body_size,
            max_clients=max_clients)

        http_client = httpclient.AsyncHTTPClient()

        # Construct the JSON Payload for each image
        json_payload = (self.format_image_payload(i) for i in images)
        payloads = (escape.json_encode(jp) for jp in json_payload)

        kwargs = {
            'method': 'POST',
            'request_timeout': timeout,
            'connect_timeout': timeout
        }

        results = []
        for payload in payloads:
            try:
                kwargs['body'] = payload
                response = await http_client.fetch(url, **kwargs)
                result = self.handle_tornado_response(response)
                results.append(result)
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
        #     self.logger.debug('Got results of shape: %s', final_prediction.shape)
        #     results.append(final_prediction)
        return results


class TensorFlowServingClient(Client):
    """Class to interact with TensorFlow Serving"""

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

    def format_image_payload(self, image):
        """Format image as JSON payload for tf-serving"""
        return {'instances': [{'image': image.tolist()}]}

    def handle_tornado_response(self, response):
        text = response.body
        try:
            prediction_json = json.loads(text)
        except:
            prediction_json = self.fix_json(text)

        final_prediction = np.array(list(prediction_json['predictions'][0]))
        return final_prediction

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


class DataProcessingClient(Client):
    """Class to interact with the DataProcessing API"""

    def get_url(self, process_type, function):
        """Get API URL for TensorFlow Serving, based on model name and version
        # Arguments:
            process_type: pre or post processing
            function: name of function to use on data
        # Returns:
            formatted URL for HTTP request
        """
        return 'http://{}:{}/{}/{}'.format(
            self.host, self.port, process_type, function)

    def post_image(self):
        pass
    
    async def tornado_images(self,
                             images,
                             model_name,
                             model_version,
                             timeout=300,
                             max_clients=10):
        """POSTs many images to the API at once using tornado.
        # Arguments:
            images: list of image data to pass to the API
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
