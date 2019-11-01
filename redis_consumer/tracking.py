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

import timeit

from deepcell_tracking import cell_tracker as _cell_tracker


class cell_tracker(_cell_tracker):
    """Override the original cell_tracker class to call model.progress()"""

    def _track_cells(self):
        """Tracks all of the cells in every frame."""
        for frame in range(1, self.x.shape[0]):
            t = timeit.default_timer()
            self.logger.info('Tracking frame %s', frame)

            cost_matrix, predictions = self._get_cost_matrix(frame)

            assignments = self._run_lap(cost_matrix)

            self._update_tracks(assignments, frame, predictions)
            self.model.progress(frame / self.x.shape[0])
            self.logger.info('Tracked frame %s in %s seconds.',
                             frame, timeit.default_timer() - t)
