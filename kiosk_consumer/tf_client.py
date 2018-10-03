# Copyright 2016-2018 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/kiosk-consumer/LICENSE
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

import logging
import json

import numpy as np
import requests
import datetime

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
    
    def fix_json(self, response):
        """Sometimes TF Serving has strange scientific notation e.g. '1e5.0,'
        so convert the float-exponent to an integer for JSON parsing.
        """
        self.logger.debug('tf-serving response is not well-formed JSON. '
                          'Attempting to fix the response.')
        try:
            raw_text = response.text
            fixed_text = raw_text.replace('.0,', ',').replace('.0],', '],')
            fixed_json = json.loads(fixed_text)
            self.logger.debug('Successfully parsed tf-serving JSON response')
            return fixed_json
        except Exception as err:
            self.logger.error('Cannot fix tf-serving response JSON: %s', err)
            raise err

    def post_image(self, image, model_name, version, timeout=300):
        """Sends image to tensorflow serving and returns response
        # Arguments:
            image: numpy array of image data passed to model
            model_name: hosted model to send image data
            version: model version to query
        # Returns: tf-serving results as numpy array    
        """
        # Define payload to send to API URL
        payload = {
            'instances': [
                {
                    'image': image.tolist()
                }
            ]
        }

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
