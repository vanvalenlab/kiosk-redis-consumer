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
from timeit import default_timer

import numpy as np
import requests


class Client(object):  # pylint: disable=useless-object-inheritance
    """Abstract Base class for API Clients"""

    def __init__(self, host, port, max_body_size=1073741824):
        self.host = host
        self.port = port
        self.max_body_size = max_body_size
        self.logger = logging.getLogger(str(self.__class__.__name__))

    def get_url(self, *args):
        """Based on the inputs, return a formatted API URL"""
        raise NotImplementedError('Override this function in a child class.')

    def format_image_payload(self, image):
        """Format the payload of the list of numpy image data.
        # Arguments:
            images: list of np arrays of image data
        # Returns:
            Formatted payload for the specific API
        """
        raise NotImplementedError('Override this function in a child class.')

    def handle_response(self, response):
        """Handle the API response.
        Each client will have a different implementation
        # Arguments:
            response: response from the http server
        # Returns:
            result: data parsed from the API repsonse
        """
        raise NotImplementedError('Override this function in a child class.')

    def _post(self, url, image, session, retries=5):
        """POST image data to given URL using async session.
        # Arguments:
            url: API endpoint URL
            image: np array of image data
            session: async session
            retries: number of retries for each request
        # Returns:
            API response parsed by self.handle_response
        """
        backoff = np.random.randint(3, 6)
        payload = self.format_image_payload(image)
        with session.post(url, json=payload) as response:
            for _ in range(retries):
                try:
                    return self.handle_response(response.json())
                except (aiohttp.ClientPayloadError,
                        aiohttp.ClientResponseError) as err:
                    self.logger.error('Encountered %s: %s. Retrying in %ss'
                                      '...', type(err).__name__, err, backoff)
            raise ValueError('Maximum retries exceeded ({})'.format(retries))

    def post_image(self, image, url, retries=5, max_clients=3):
        """POSTs many images to the API at once.
        # Arguments:
            image: image data to pass to the API
            url: URL to send the request
            max_clients: max number of simultaneous http clients
            retries: max number of attempts to retry the POST request
        # Returns:
            results: list of results from API
        """
        conn = aiohttp.TCPConnector(limit=max_clients)
        with aiohttp.ClientSession(connector=conn) as session:
            request = self._post(url, image, session, retries=retries)
            return request

    def verify_endpoint_liveness(self,
                                 num_retries=60,
                                 timeout=10,
                                 retry_interval=10,
                                 code=200,
                                 endpoint=None):
        """Ping API to check if the service is up.
        # Arguments:
            num_retries: number of attempts to ping
            timeout: timeout in seconds for each ping attempt
            retry_interval: time to wait between retries
            code: expected healthy HTTP response code
            endpoint: the health-check endpoint of the host
        # Returns:
            True if service is live otherwise False
        """
        liveness_url = 'http://{}:{}'.format(self.host, self.port)
        if endpoint is not None:
            liveness_url = '{}/{}'.format(liveness_url, endpoint)

        for i in range(num_retries):
            try:
                response = requests.get(liveness_url, timeout=timeout)
                if response.status_code == code:
                    self.logger.debug('Connection established after '
                                      '%s attempts.', i + 1)
                    break

                self.logger.error('Expected a %s response but got %s. '
                                  'Entered `unreachable` code block.',
                                  code, response.status_code)

            except requests.exceptions.ConnectionError as err:
                self.logger.warning('Encountered %s while checking API '
                                    ' liveness:  %s', type(err).__name__, err)
            except Exception as err:
                self.logger.warning('Encountered %s while checking API '
                                    'liveness: %s', type(err).__name__, err)
                raise err
            # sleep as long as needed to allow tf-serving time to startup
            time.sleep(retry_interval)
        else:  # for/else loop.  only enters block after all retries
            self.logger.error('Connection to tf-serving not established. '
                              'Exhausted all %s retries.', num_retries)
            return False
        return True


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

    def format_image_payload(self, image):
        """Format the payload of the list of numpy image data.
        # Arguments:
            images: list of np arrays of image data
        # Returns:
            Formatted payload for the specific API
        """
        start = default_timer()
        self.logger.debug('JSON formatting image of shape %s', image.shape)
        try:
            payload = {'instances': [{'image': image.tolist()}]}
            self.logger.debug('Successfully JSON formatted image payload of '
                              'shape %s in %ss', image.shape,
                              default_timer() - start)
            return payload
        except Exception as err:
            self.logger.error('Failed to JSON format image of shape %s due to '
                              '%s: %s', image.shape, type(err).__name__, err)
            raise err

    def handle_response(self, response):
        """Handle the API response.
        Each client will have a different implementation
        # Arguments:
            response: response from the http server
        # Returns:
            result: data parsed from the API repsonse
        """
        # try:
        #     prediction_json = json.loads(response.body)
        # except:  # pylint: disable=bare-except
        #     prediction_json = self.fix_json(text)
        result = np.array(list(response['predictions'][0]))
        self.logger.debug('Loaded response into np.array of shape %s',
                          result.shape)
        return result

    def fix_json(self, response_text):
        """Some TensorFlow Serving versions have strange scientific notation
        bug (e.g. '1e5.0,'). Convert the float exponent to an int.
        # Arguments:
            response_text: HTTP response as string
        # Returns:
            fixed_json: HTTP response as JSON object
        """
        self.logger.debug('tf-serving response is not well-formed JSON. '
                          'Attempting to fix the response.')
        try:
            fixed = response_text.replace('.0,', ',').replace('.0],', '],')
            fixed_json = json.loads(fixed)
            self.logger.debug('Successfully parsed tf-serving JSON response')
            return fixed_json
        except Exception as err:
            self.logger.error('Cannot fix tf-serving response JSON: %s', err)
            raise err
