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
"""PolarisConsumer class for consuming Polaris jobs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import timeit

import matplotlib.pyplot as plt
import numpy as np

from deepcell_spots.applications import Polaris

from redis_consumer.consumers import TensorFlowServingConsumer
from redis_consumer import settings
from redis_consumer import utils


class PolarisConsumer(TensorFlowServingConsumer):
    """Consumes image files and uploads the results"""

    def save_output(self, coords, image, save_name):
        """Save output in a zip file and upload it. Output includes predicted spot locations
        plotted on original image as a .tiff file and coordinate spot locations as .npy file"""
        with tempfile.TemporaryDirectory() as tempdir:
            # Save each result channel as an image file
            subdir = os.path.dirname(save_name.replace(tempdir, ''))
            name = os.path.splitext(os.path.basename(save_name))[0]

            outpaths = []
            for i in range(len(coords)):
                # Save image with plotted spot locations
                img_name = '{}.tif'.format(i)
                if name:
                    img_name = '{}_{}'.format(name, img_name)

                img_path = os.path.join(tempdir, subdir, img_name)

                fig = plt.figure()
                plt.ioff()
                plt.imshow(image[i], cmap='gray')
                plt.scatter(coords[i][:, 1], coords[i][:, 0], c='m', s=6)
                plt.xticks([])
                plt.yticks([])
                plt.savefig(img_path)

                # Save coordiates
                coords_name = '{}.npy'.format(i)
                if name:
                    coords_name = '{}_{}'.format(name, coords_name)

                coords_path = os.path.join(tempdir, subdir, coords_name)

                np.save(coords_path, coords[i])

                outpaths.extend([img_path, coords_path])
                # outpaths.extend([coords_path])

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

        segmentation_type = hvals.get('segmentation_type')

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

        # Get model_name and version
        spots_model_name, spots_model_version = settings.POLARIS_MODEL.split(':')

        # detect dimension order and add to redis
        dim_order = self.detect_dimension_order(image, spots_model_name, spots_model_version)
        self.update_key(redis_hash, {
            'dim_order': ','.join(dim_order)
        })

        spots_image = self.validate_model_input(image[..., 0:1],
                                                spots_model_name,
                                                spots_model_version)

        if segmentation_type == 'tissue':
            compartment = 'mesmer'
            seg_model_name, seg_model_version = settings.MESMER_MODEL.split(':')
            segmentation_model = self.get_model_wrapper(settings.MESMER_MODEL,
                                                        batch_size=1)

            seg_image = self.validate_model_input(image[..., 1:],
                                                  seg_model_name,
                                                  seg_model_version)

        elif segmentation_type == 'cell culture':
            channels = hvals.get('channels').split(',')
            if channels[1] == '':
                compartment = 'cytoplasm'
                seg_model_name, seg_model_version = settings.MODEL_CHOICES[2].split(':')
                segmentation_model = self.get_model_wrapper(settings.MODEL_CHOICES[2],
                                                            batch_size=1)
            elif channels[2] == '':
                compartment = 'nucleus'
                seg_model_name, seg_model_version = settings.MODEL_CHOICES[0].split(':')
                segmentation_model = self.get_model_wrapper(settings.MODEL_CHOICES[0],
                                                            batch_size=1)

            seg_image = self.validate_model_input(image[..., 1:2],
                                                  seg_model_name,
                                                  seg_model_version)

        else:
            compartment = 'no segmentation'
            segmentation_model = None

        # Send data to the model
        self.update_key(redis_hash, {'status': 'predicting'})

        app = self.get_grpc_app(settings.POLARIS_MODEL, Polaris)

        # with new batching update in deepcell.applications,
        # app.predict() cannot handle a batch_size of None.
        threshold = hvals.get('threshold', settings.POLARIS_THRESHOLD)
        clip = hvals.get('clip', settings.POLARIS_CLIP)
        results = app.predict(spots_image=spots_image, segmentation_image=seg_image,
                              batch_size=4, threshold=threshold, clip=clip)

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
