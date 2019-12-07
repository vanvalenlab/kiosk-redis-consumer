# Copyright 2016-2019 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
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
# ==============================================================================
"""Override the deepcell_tracking.CellTracker class to update progress"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import timeit

from deepcell_tracking import CellTracker as _CellTracker


class CellTracker(_CellTracker):
    """Override the original cell_tracker class to call model.progress()"""

    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(str(self.__class__.__name__))
        super(CellTracker, self).__init__(*args, **kwargs)

    def track_cells(self):
        """Tracks all of the cells in every frame.
        """
        start = timeit.default_timer()
        self._initialize_tracks()

        for frame in range(1, self.x.shape[self.time_axis]):
            self._track_frame(frame)

            # The only difference between the original and this
            # is calling model.progress after every frame.
            self.model.progress(frame / self.x.shape[0])

        self.logger.info('Tracked all %s frames in %s s.',
                         self.x.shape[self.time_axis],
                         timeit.default_timer() - start)
