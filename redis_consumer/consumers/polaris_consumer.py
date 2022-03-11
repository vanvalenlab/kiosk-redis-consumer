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
"""PolarisConsumer class for consuming singleplex FISH analysis jobs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import os
import tempfile
import time
import timeit
import uuid

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from deepcell_spots.singleplex import match_spots_to_cells

from redis_consumer.consumers import TensorFlowServingConsumer
from redis_consumer import settings
from redis_consumer import utils

class PolarisConsumer(TensorFlowServingConsumer):
    """Consumes multichannnel image files for singleplex FISH analysis, adds single 
    channel images to spot detection and segmentation queues, and uploads the results
    """

    def _add_images(self, hvals, uid, image, queue, label=''):
        """
        Uploads image to a temporary directory and adds it to the redis queue
        for analysis.
        """
        with tempfile.TemporaryDirectory() as tempdir:
            # Save and upload the spots image
            image_fname = '{}-{}-{}-image.tif'.format(
                uid, hvals.get('original_name'), queue)
            image_local_path = os.path.join(tempdir, image_fname)
            tifffile.imsave(image_local_path, image)
            upload_file_name, upload_file_url = self.storage.upload(
                image_local_path)

        self.logger.debug('Image shape: {}'.format(image.shape))
        # prepare hvals for this images's hash
        current_timestamp = self.get_current_timestamp()
        channel_str = '0,1' if queue=='mesmer' else '0'
        image_hvals = {
            'identity_upload': self.name,
            'input_file_name': upload_file_name,
            'original_name': image_fname,
            'status': 'new',
            'created_at': current_timestamp,
            'updated_at': current_timestamp,
            'url': upload_file_url,
            'channels': channel_str,
            'label': label
            }

        # make a hash for this frame
        image_hash = '{prefix}:{file}:{hash}'.format(
            prefix=queue,
            file=image_fname,
            hash=uuid.uuid4().hex)

        self.redis.hmset(image_hash, image_hvals)
        self.redis.lpush(queue, image_hash)
        self.logger.debug('Added new hash to %s queue `%s`: %s',
                          queue, image_hash, json.dumps(image_hvals, indent=4))

        return(image_hash)

    def _analyze_images(self, redis_hash, subdir, fname):
        """
        Given the upload location `input_file_name`, and the downloaded
        location of the same file in subdir/fname, return the raw and annotated
        data.
        """
        hvals = self.redis.hgetall(redis_hash)
        raw = utils.get_image(os.path.join(subdir, fname))

        # remove the last dimensions added by `get_image`
        tiff_stack = np.squeeze(raw, -1)

        self.logger.debug('Got tiffstack shape %s.', tiff_stack.shape)

        # get segmentation type and channel order
        channels = hvals.get('channels').split(',')
        self.logger.debug('Channels: {}'.format(hvals.get('channels')))
        segmentation_type = hvals.get('segmentation_type')

        remaining_hashes = set()
        uid = uuid.uuid4().hex

        self.logger.debug('Starting spot detection')
        # get spots image and add to spot_detection queue
        spots_image = tiff_stack[..., int(channels[0])]
        self.logger.debug('Spot image size: {}'.format(spots_image.shape))
        spots_hash = self._add_images(hvals, uid, spots_image, queue='spot')
        remaining_hashes.add(spots_hash)

        self.logger.debug('Starting segmentation')
        if segmentation_type == 'cell culture':
            if channels[1]:
                # add channel 1 ind of tiff stack to segmentation queue
                cyto_image = tiff_stack[..., int(channels[1])]
                cyto_hash = self._add_images(hvals, uid, cyto_image, queue='segmentation', label='2')
                remaining_hashes.add(cyto_hash)

            if channels[2]:
                # add channel 2 ind of tiff stack to nuclear queue
                nuc_image = tiff_stack[..., int(channels[2])]
                nuc_hash = self._add_images(hvals, uid, nuc_image, queue='segmentation', label='0')
                remaining_hashes.add(nuc_hash)

        elif segmentation_type == 'tissue':
            # add ims 1 and 2 to mesmer queue
            mesmer_image = np.concat((tiff_stack[..., int(channels[1])], 
                                      tiff_stack[..., int(channels[2])]), axis=-1)
            mesmer_hash = self._add_images(hvals, uid, mesmer_image, queue='mesmer')
            remaining_hashes.add(mesmer_hash)

        coords = []
        segmentation_results = []
        while remaining_hashes:
            finished_hashes = set()
            for h in remaining_hashes:
                status = self.redis.hget(h, 'status')

                self.logger.debug('Hash %s has status %s',
                                  h, status)

                if status == self.failed_status:
                    # Analysis failed
                    reason = self.redis.hget(h, 'reason')
                    raise RuntimeError(
                        'Analysis failed for image with hash: {} '
                        'for this reason: {}'.format(h, reason))

                if status == self.final_status:
                    # Analysis finished
                    with tempfile.TemporaryDirectory() as tempdir:
                        out = self.redis.hget(h, 'output_file_name')
                        pred_zip = self.storage.download(out, tempdir)
                        pred_files = list(utils.iter_image_archive(
                            pred_zip, tempdir))

                        if 'spot' in h:
                            # handle spot detection results
                            for i,file in enumerate(pred_files):
                                if file.endswith('.npy'):
                                    spots_pred = np.load(pred_files[i])
                                    coords.append(spots_pred)
                        else:
                            # handle segmentation results
                            for i, file in enumerate(pred_files):
                                seg_pred = utils.get_image(pred_files[i])
                                segmentation_results.append(seg_pred)

                        finished_hashes.add(h)

            remaining_hashes -= finished_hashes
            time.sleep(settings.INTERVAL)

        return {'coords': np.array(coords), 'segmentation': segmentation_results}

    def _consume(self, redis_hash):
        start = timeit.default_timer()
        hvals = self.redis.hgetall(redis_hash)

        if hvals.get('status') in self.finished_statuses:
            self.logger.warning('Found completed hash `%s` with status %s.',
                                redis_hash, hvals.get('status'))
            return hvals.get('status')

        self.update_key(redis_hash, {
            'status': 'started',
            'identity_started': self.name,
        })

        with tempfile.TemporaryDirectory() as tempdir:
            fname = self.storage.download(hvals.get('input_file_name'),
                                          tempdir)
            res = self._analyze_images(redis_hash, tempdir, fname)

        self.logger.debug('Finished spot detection and segmentation.')
        self.logger.debug('Coords shape: %s', np.shape(res['coords']))
        self.logger.debug('Segmentation result shape: %s', np.shape(res['segmentation']))

        # Assign spots to cells
        coords = np.array(res['coords'])
        labeled_im = np.squeeze(np.array(res['segmentation']), axis=0)
        fname = hvals.get('input_file_name')
        image = self.download_image(fname)
        channels = hvals.get('channels').split(',')
        spots_image = image[..., int(channels[0])]

        save_name = hvals.get('original_name', fname)

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

                fig,ax = plt.subplots(1, 2, figsize=(16,8))
                plt.ioff()
                ax[0].imshow(spots_image[i], cmap='gray')
                ax[0].scatter(coords[i][:, 1], coords[i][:, 0], c='m', s=6)
                ax[1].imshow(labeled_im[i,:,:,0], cmap='jet')
                
                for ii in range(len(ax)):
                    ax[ii].set_xticks([])
                    ax[ii].set_yticks([])
                plt.savefig(img_path)

                # Save coordinates
                spot_dict = match_spots_to_cells(np.expand_dims(labeled_im[i], axis=0), coords[i])
                
                csv_name = '{}.csv'.format(i)
                if name:
                    csv_name = '{}_{}'.format(name, csv_name)
                csv_path = os.path.join(tempdir, subdir, csv_name)
                csv_header = ['x', 'y', 'cellID']
                with open(csv_path, 'w', newline='') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',')
                    writer.writerow(csv_header)
                    for key in spot_dict.keys():
                        cell_coords = spot_dict[key]
                        for i in range(len(cell_coords)):
                            loc = cell_coords[i]
                            writer.writerow([loc[0], loc[1], key])

                outpaths.extend([img_path, csv_path])

            # Save each prediction image as zip file
            zip_file = utils.zip_files(outpaths, tempdir)
        
            # Upload the zip file to cloud storage bucket
            cleaned = zip_file.replace(tempdir, '')
            subdir = os.path.dirname(utils.strip_bucket_path(cleaned))
            subdir = subdir if subdir else None
            dest, output_url = self.storage.upload(zip_file, subdir=subdir)

        # Update redis with the final results
        end = timeit.default_timer()
        self.update_key(redis_hash, {
            'status': self.final_status,
            'output_url': output_url,
            'upload_time': end, # - _
            'output_file_name': dest,
            'total_jobs': 1,
            'total_time': end - start,
            'finished_at': self.get_current_timestamp()
        })
        return self.final_status