# Copyright 2016-2022 The Van Valen Lab at the California Institute of
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
"""MesmerConsumer class for consuming Mesmer whole cell segmentation jobs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import timeit

import numpy as np

from deepcell_spots.applications import Polaris

from redis_consumer.consumers import TensorFlowServingConsumer
from redis_consumer import settings


class PolarisConsumer(TensorFlowServingConsumer):
    """Consumes image files and uploads the results"""

    def save_output(self, coords, image, save_name):
        """Save output images into a zip file and upload it."""
        with tempfile.TemporaryDirectory() as tempdir:
            # Save each result channel as an image file
            subdir = os.path.dirname(save_name.replace(tempdir, ''))
            name = os.path.splitext(os.path.basename(save_name))[0]

            if not isinstance(coords, list):
                coords = [coords]

            outpaths = []
            for i in range(len(coords)):
                # Save image with plotted spot locations
                img_name = '{}.tif'.format(i)
                if name:
                    img_name = '{}_{}'.format(name, img_name)

                img_path = os.path.join(output_dir, subdir, img_name)

                fig = plt.figure()
                plt.ion()
                plt.imshow(image[i])
                plt.scatter(coords[i][:,1], coords[0][:,0], edgecolors='r', facecolors='None')
                plt.xticks([])
                plt.yticks([])
                plt.close(fig)
                plt.savefig(img_path)

                # Save coordiates
                coords_name = '{}.txt'.format(i)
                if name:
                    coords_name = '{}_{}'.format(name, coords_name)

                coords_path = os.path.join(output_dir, subdir, coords_name)
                
                np.savetxt(coords_path, coords[i])

                outpaths.extend([img_path, coords_path])

            # Save each prediction image as zip file
            zip_file = utils.zip_files(outpaths, tempdir)

            # Upload the zip file to cloud storage bucket
            cleaned = zip_file.replace(tempdir, '')
            subdir = os.path.dirname(utils.strip_bucket_path(cleaned))
            subdir = subdir if subdir else None
            dest, output_url = self.storage.upload(zip_file, subdir=subdir)

        return dest, output_url

    def _consume(self, redis_hash):
        start = timeit.default_timer()
        hvals = self.redis.hgetall(redis_hash)

        if hvals.get('status') in self.finished_statuses:
            self.logger.warning('Found completed hash `%s` with status %s.',
                                redis_hash, hvals.get('status'))
            return hvals.get('status')

        self.logger.debug('Found hash to process `%s` with status `%s`.',
                          redis_hash, hvals.get('status'))

        self.update_key(redis_hash, {
            'status': 'started',
            'identity_started': self.name,
        })

        # Get model_name and version
        model_name, model_version = settings.POLARIS_MODEL.split(':')

        _ = timeit.default_timer()

        # Load input image
        fname = hvals.get('input_file_name')
        image = self.download_image(fname)

        # squeeze extra dimension that is added by get_image
        image = np.squeeze(image)
        # add in the batch dim
        image = np.expand_dims(image, axis=0)

        # Pre-process data before sending to the model
        self.update_key(redis_hash, {
            'status': 'pre-processing',
            'download_time': timeit.default_timer() - _,
        })

        scale = 1

        # detect dimension order and add to redis
        dim_order = self.detect_dimension_order(image, model_name, model_version)
        self.update_key(redis_hash, {
            'dim_order': ','.join(dim_order)
        })

        # Validate input image
        if hvals.get('channels'):
            channels = [int(c) for c in hvals.get('channels').split(',')]
        else:
            channels = None

        image = self.validate_model_input(image, model_name, model_version,
                                          channels=channels)

        # Send data to the model
        self.update_key(redis_hash, {'status': 'predicting'})

        app = self.get_grpc_app(settings.POLARIS_MODEL, Polaris)

        # with new batching update in deepcell.applications,
        # app.predict() cannot handle a batch_size of None.
        batch_size = app.model.get_batch_size()
        results = app.predict(image, batch_size=batch_size)

        # Save the post-processed results to a file
        _ = timeit.default_timer()
        self.update_key(redis_hash, {'status': 'saving-results'})

        save_name = hvals.get('original_name', fname)
        dest, output_url = self.save_output(results, image, save_name)

        # Update redis with the final results
        end = timeit.default_timer()
        self.update_key(redis_hash, {
            'status': self.final_status,
            'output_url': output_url,
            'upload_time': end - _,
            'output_file_name': dest,
            'total_jobs': 1,
            'total_time': end - start,
            'finished_at': self.get_current_timestamp()
        })
        return self.final_status