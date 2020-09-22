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
import numpy as np

from redis_consumer.processing import multiplex_postprocess_consumer


# return input dicts to make sure they were passed appropriately
def mocked_postprocessing(model_output, compartment, whole_cell_kwargs, nuclear_kwargs):
    return whole_cell_kwargs, nuclear_kwargs


def test_multiplex_postprocess_consumer(mocker):
    mocker.patch('redis_consumer.processing.multiplex_postprocess', mocked_postprocessing)

    model_output = np.zeros((1, 40, 40, 4))
    compartment = 'both'

    defalt_cell_dict = {'maxima_threshold': 0.1, 'maxima_model_smooth': 0,
                        'interior_threshold': 0.3, 'interior_model_smooth': 2,
                        'small_objects_threshold': 15,
                        'fill_holes_threshold': 15,
                        'radius': 2}

    default_nuc_dict = {'maxima_threshold': 0.1, 'maxima_model_smooth': 0,
                        'interior_threshold': 0.6, 'interior_model_smooth': 0,
                        'small_objects_threshold': 15,
                        'fill_holes_threshold': 15,
                        'radius': 2}

    cell_dict, nuc_dict = multiplex_postprocess_consumer(model_output=model_output,
                                                         compartment=compartment,
                                                         whole_cell_kwargs=None,
                                                         nuclear_kwargs=None)

    assert defalt_cell_dict == cell_dict
    assert default_nuc_dict == nuc_dict

    modified_cell_dict = {'maxima_threshold': 0.4, 'maxima_model_smooth': 4,
                          'small_objects_threshold': 2,
                          'radius': 0}

    modified_nuc_dict = {'maxima_threshold': 0.43, 'maxima_model_smooth': 41,
                         'small_objects_threshold': 20,
                         'radius': 4}

    cell_dict, nuc_dict = multiplex_postprocess_consumer(model_output=model_output,
                                                         compartment=compartment,
                                                         whole_cell_kwargs=modified_cell_dict,
                                                         nuclear_kwargs=modified_nuc_dict)

    assert modified_cell_dict == cell_dict
    assert modified_nuc_dict == nuc_dict
