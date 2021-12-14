# Copyright 2016-2021 The Van Valen Lab at the California Institute of
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
"""TrackingConsumer class for consuming cell tracking jobs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tempfile
import time
import timeit
import uuid

import numpy as np
import tifffile

from deepcell_toolbox.processing import correct_drift

from deepcell.applications import CellTracking

from redis_consumer.consumers import TensorFlowServingConsumer
from redis_consumer import utils
from redis_consumer import settings


class TrackingConsumer(TensorFlowServingConsumer):
    """Consumes some unspecified file format, tracks the images,
       and uploads the results
    """

    def is_valid_hash(self, redis_hash):
        if redis_hash is None:
            return False

        fname = str(self.redis.hget(redis_hash, 'input_file_name')).lower()

        valid_file = (fname.endswith('.trk') or
                      fname.endswith('.trks') or
                      fname.endswith('.tif') or
                      fname.endswith('.tiff'))

        self.logger.debug('Got key %s and decided %s', redis_hash, valid_file)

        return valid_file

    def _load_data(self, redis_hash, subdir, fname):
        """
        Given the upload location `input_file_name`, and the downloaded
        location of the same file in subdir/fname, return the raw and annotated
        data. If the input is only raw data, we call up the ImageFileConsumer
        to predict and annotate the data.

        Args:
            hvalues: map of original hvalues of the tracking job
            subdir: string of path that contains the downloaded file
            fname: string of file name inside subdir
        """
        hvalues = self.redis.hgetall(redis_hash)

        if fname.endswith('.trk') or fname.endswith('.trks'):
            return utils.load_track_file(os.path.join(subdir, fname))

        if not fname.endswith('.tiff') and not fname.endswith('.tif'):
            raise ValueError('_load_data takes in only .tiff, .trk, or .trks')

        # push a key per frame and let ImageFileConsumers segment
        tiff_stack = utils.get_image(os.path.join(subdir, fname))

        # remove the last dimensions added by `get_image`
        # tiff_stack = np.squeeze(tiff_stack, -1)
        if len(tiff_stack.shape) != 3:
            raise ValueError('This tiff file has shape {}, which is not 3 '
                             'dimensions. Tracking can only be done on images '
                             'with 3 dimensions, (time, width, height)'.format(
                                 tiff_stack.shape))

        num_frames = len(tiff_stack)
        hash_to_frame = {}
        remaining_hashes = set()
        frames = {}

        if num_frames > settings.MAX_IMAGE_FRAMES:
            raise ValueError('This tiff file has {} frames but the maximum '
                             'number of allowed frames is {}.'.format(
                                 num_frames, settings.MAX_IMAGE_FRAMES))

        self.logger.debug('Got tiffstack shape %s.', tiff_stack.shape)

        uid = uuid.uuid4().hex
        for i, img in enumerate(tiff_stack):

            with tempfile.TemporaryDirectory() as tempdir:
                # Save and upload the frame.
                segment_fname = '{}-{}-tracking-frame-{}.tif'.format(
                    uid, hvalues.get('original_name'), i)
                segment_local_path = os.path.join(tempdir, segment_fname)
                tifffile.imsave(segment_local_path, img)
                upload_file_name, upload_file_url = self.storage.upload(
                    segment_local_path)

            # prepare hvalues for this frame's hash
            current_timestamp = self.get_current_timestamp()
            frame_hvalues = {
                'identity_upload': self.name,
                'input_file_name': upload_file_name,
                'original_name': segment_fname,
                'status': 'new',
                'created_at': current_timestamp,
                'updated_at': current_timestamp,
                'url': upload_file_url,
                'channels': hvalues.get('channels', ''),
            }

            # make a hash for this frame
            segment_hash = '{prefix}:{file}:{hash}'.format(
                prefix=settings.SEGMENTATION_QUEUE,
                file=segment_fname,
                hash=uuid.uuid4().hex)

            # push the hash to redis and the predict queue
            self.redis.hmset(segment_hash, frame_hvalues)
            self.redis.lpush(settings.SEGMENTATION_QUEUE, segment_hash)
            self.logger.debug('Added new hash for segmentation `%s`: %s',
                              segment_hash, json.dumps(frame_hvalues, indent=4))
            hash_to_frame[segment_hash] = i
            remaining_hashes.add(segment_hash)

        # pop hash, check it, and push it back if it's not done
        # this checks the same hash over and over again, since set's
        # pop is not random. This is fine, since we still need every
        # hash to finish before doing anything.
        while remaining_hashes:
            finished_hashes = set()
            for segment_hash in remaining_hashes:
                status = self.redis.hget(segment_hash, 'status')

                self.logger.debug('Hash %s has status %s',
                                  segment_hash, status)

                if status == self.failed_status:
                    # Segmentation failed, tracking cannot be finished.
                    reason = self.redis.hget(segment_hash, 'reason')
                    raise RuntimeError(
                        'Tracking failed during segmentation on frame {}. '
                        'Segmentation Error: {}'.format(
                            hash_to_frame[segment_hash], reason))

                if status == self.final_status:
                    # Segmentation is finished, save and load the frame.
                    with tempfile.TemporaryDirectory() as tempdir:
                        out = self.redis.hget(segment_hash, 'output_file_name')
                        frame_zip = self.storage.download(out, tempdir)
                        frame_files = list(utils.iter_image_archive(
                            frame_zip, tempdir))

                        if len(frame_files) != 1:
                            raise RuntimeError(
                                'After unzipping predicted frame, got '
                                'back multiple files {}. Expected a '
                                'single file.'.format(frame_files))

                        frame_idx = hash_to_frame[segment_hash]
                        frames[frame_idx] = utils.get_image(frame_files[0])
                        finished_hashes.add(segment_hash)

            remaining_hashes -= finished_hashes
            time.sleep(settings.INTERVAL)

        labels = [frames[i] for i in range(num_frames)]

        # Cast y to int to avoid issues during fourier transform/drift correction
        y = np.array(labels, dtype='uint16')
        # TODO: Why is there an extra dimension?
        # Not a problem in tests, only with application based results.
        # Issue with batch dimension from outputs?
        y = y[:, 0] if y.shape[1] == 1 else y
        return {'X': tiff_stack, 'y': y}

    def _consume(self, redis_hash):
        start = timeit.default_timer()
        hvals = self.redis.hgetall(redis_hash)

        if hvals.get('status') in self.finished_statuses:
            self.logger.warning('Found completed hash `%s` with status %s.',
                                redis_hash, hvals.get('status'))
            return hvals.get('status')

        # Set status and initial progress
        self.update_key(redis_hash, {
            'status': 'started',
            'progress': 0,
            'identity_started': self.name,
        })

        with tempfile.TemporaryDirectory() as tempdir:
            fname = self.storage.download(hvals.get('input_file_name'),
                                          tempdir)
            data = self._load_data(redis_hash, tempdir, fname)

        self.logger.debug('Got contents tracking file contents.')
        self.logger.debug('X shape: %s', data['X'].shape)
        self.logger.debug('y shape: %s', data['y'].shape)

        # Correct for drift if enabled
        if settings.DRIFT_CORRECT_ENABLED:
            t = timeit.default_timer()
            data['X'], data['y'] = correct_drift(data['X'], data['y'])
            self.logger.debug('Drift correction complete in %s seconds.',
                              timeit.default_timer() - t)

        # Prep Neighborhood_Encoder
        neighborhood_encoder = self.get_model_wrapper(settings.NEIGHBORHOOD_ENCODER,
                                                      batch_size=64)

        # Send data to the model
        app = self.get_grpc_app(settings.TRACKING_MODEL, CellTracking,
                                neighborhood_encoder=neighborhood_encoder,
                                birth=settings.BIRTH,
                                death=settings.DEATH,
                                division=settings.DIVISION,
                                track_length=settings.TRACK_LENGTH,
                                embedding_axis=1)

        self.logger.debug('Tracking...')
        self.update_key(redis_hash, {'status': 'predicting'})
        results = app.predict(data['X'], data['y'])
        self.logger.debug('Tracking done!')

        self.update_key(redis_hash, {'status': 'saving-results'})
        with tempfile.TemporaryDirectory() as tempdir:
            # Save lineage data to JSON file
            lineage_file = os.path.join(tempdir, 'lineage.json')
            with open(lineage_file, 'w') as fp:
                json.dump(results['tracks'], fp)

            save_name = hvals.get('original_name', fname)
            subdir = os.path.dirname(save_name.replace(tempdir, ''))
            name = os.path.splitext(os.path.basename(save_name))[0]

            # Save tracked data as tiff stack
            outpaths = utils.save_numpy_array(
                results['y_tracked'], name=name,
                subdir=subdir, output_dir=tempdir)

            outpaths.append(lineage_file)

            # Save as zip instead of .trk for demo-purposes
            zip_file = utils.zip_files(outpaths, tempdir)

            output_file_name, output_url = self.storage.upload(zip_file)

        t = timeit.default_timer() - start
        self.update_key(redis_hash, {
            'status': self.final_status,
            'output_url': output_url,
            'output_file_name': output_file_name,
            'finished_at': self.get_current_timestamp(),
            'total_jobs': 1,
            'total_time': t,
        })
        return self.final_status
