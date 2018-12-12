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
import asyncio
import logging
import time

import numpy as np
import requests
import aiohttp
import tornado
from tornado import httpclient


class Client(object):  # pylint: disable=useless-object-inheritance
    """Abstract Base class for API Clients"""

    def __init__(self, host, port, max_body_size=1073741824):
        self.host = host
        self.port = port
        self.max_body_size = max_body_size
        self.logger = logging.getLogger(str(self.__class__.__name__))

    def get_url(self, *args):
        """Based on the inputs, return a formatted API URL"""
        raise NotImplementedError

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
                self.logger.warning('Encountered unexpected %s while checking '
                                    'API liveness: %s', type(err).__name__, err)
                raise err
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
        try:
            # downgrade the float precision to conserve memory
            self.logger.debug('Formatting image payload with shape %s as JSON',
                              image.shape)
            payload = {'instances': [{'image': image.tolist()}]}
        except Exception as err:  # pylint: disable=broad-except
            self.logger.error('Failed to format payload image with shape %s '
                              'due to %s: %s', image.shape,
                              type(err).__name__, image.shape)
        self.logger.debug('Successfully formatted image payload with shape %s'
                          ' as JSON', image.shape)
        return payload

    def handle_response(self, response):
        """Handle the API response.
        Each client will have a different implementation
        # Arguments:
            response: response from the http server
        # Returns:
            result: data parsed from the API repsonse
        """
        raise NotImplementedError

    def parse_error(self, error):
        """Parse the error message from the object.
        Override-able for various API response formats
        # Arguments:
            error: the error object
        # Returns:
            the error message as a string
        """
        return tornado.escape.json_decode(error.response.body)['error']

    async def post_image(self,
                         image,
                         url,
                         timeout=30,
                         max_clients=10,
                         retries=5):
        """POSTs many images to the API at once.
        # Arguments:
            image: image data to pass to the API
            url: URL to send the request
            max_clients: max number of simultaneous http clients
            retries: max number of attempts to retry the POST request
        # Returns:
            results: list of results from API
        """
        backoff = np.random.randint(3, 6)
        payload = self.format_image_payload(image)
        async with aiohttp.ClientSession() as session:
            for _ in range(retries):
                try:
                    async with session.post(url, json=payload) as resp:
                        result = self.handle_response(await resp.json())
                        return result
                except (aiohttp.ClientPayloadError,
                        aiohttp.ClientResponseError) as err:
                    self.logger.error('Encountered %s: %s. Sleeping for %ss '
                                      'and retrying...', type(err).__name__,
                                      err, backoff)
                except Exception as err:
                    self.logger.error('%s: %s', type(err).__name__, err)
                    raise err
                await asyncio.sleep(backoff)
            raise ValueError('Maximum retries exceeded ({})'.format(retries))

    async def post_images(self, images, url, max_clients=10):
        """POSTs many images to the API at once.
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
            max_buffer_size=self.max_body_size,
            max_clients=max_clients)

        kwargs = {
            'method': 'POST',
            'request_timeout': 30 * len(images),
            'connect_timeout': 30 * len(images)
        }

        # Construct the JSON Payload for each image
        json_payload = (self.format_image_payload(i) for i in images)
        payloads = (tornado.escape.json_encode(jp) for jp in json_payload)
        http_client = httpclient.AsyncHTTPClient()

        results = []
        # for i, payload in enumerate(payloads):
        #     try:
        #         if payload is None:
        #             raise ZeroDivisionError
        #         response = await http_client.fetch(url, method='POST', body=payload)
        #         self.logger.info('Waited for response %s', i)
        #         result = self.handle_response(response)
        #         results.append(result)
        #     except tornado.iostream.StreamClosedError as err:
        #         self.logger.warning('Stream Closed: %s: %s', type(err).__name__, err)
        #     except httpclient.HTTPError as err:
        #         errtxt = self.parse_error(err)
        #         self.logger.error('%s %s: %s', type(err).__name__, err, errtxt)
        #         raise httpclient.HTTPError(err.code, '{}'.format(errtxt))
        #     except Exception as err:
        #         self.logger.error('%s: %s', type(err).__name__, err)
        #         raise err

        # Using tornado.gen.multi - too many requests causes tf-serving OOM.
        try:
            http_client = httpclient.AsyncHTTPClient()
            reqs = [http_client.fetch(url, body=p, **kwargs) for p in payloads]
            responses = await tornado.gen.multi(reqs)
            results = [self.handle_response(r) for r in responses]
        except tornado.iostream.StreamClosedError as err:
            self.logger.warning('Stream Closed: %s: %s', type(err).__name__, err)
        except httpclient.HTTPError as err:
            errtxt = self.parse_error(err)
            self.logger.error('%s %s: %s', type(err).__name__, err, errtxt)
            raise httpclient.HTTPError(err.code, '{}'.format(errtxt))
        except Exception as err:
            self.logger.error('%s: %s', type(err).__name__, err)
            raise err

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
        """Some TensorFlow Serving versions have strange scientific notation bug
        (e.g. '1e5.0,'). Convert the float exponent to an int for JSON parsing.
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

    def handle_response(self, response):
        # text = response.body
        # if text is None:
        #     self.logger.error('response body is None')
        # try:
        #     prediction_json = json.loads(text)
        # except:  # pylint: disable=bare-except
        #     prediction_json = self.fix_json(text)
        prediction_json = response
        result = np.array(list(prediction_json['predictions'][0]), dtype='float16')
        self.logger.debug('Loaded response into np.array of shape %s',
                          result.shape)
        return result


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
        return 'http://{}:{}/process/{}/{}'.format(
            self.host, self.port, process_type, function)

    def handle_response(self, response):
        # text = response.body
        # processed_json = json.loads(text)
        processed_json = response
        result = np.array(list(processed_json['processed'][0]), dtype='float16')
        self.logger.debug('Loaded response into np.array of shape %s',
                          result.shape)
        return result
