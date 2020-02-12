# Copyright 2016-2019 The Van Valen Lab at the California Institute of
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
"""Classes to consume events in redis"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time
import timeit
import uuid

from skimage.external import tifffile
import numpy as np

from redis_consumer.grpc_clients import TrackingClient
from redis_consumer.consumers import TensorFlowServingConsumer
from redis_consumer import utils
from redis_consumer import tracking
from redis_consumer import settings
from redis_consumer import processing


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

    def _get_model(self, redis_hash, hvalues):
        hostname = '{}:{}'.format(settings.TF_HOST, settings.TF_PORT)

        # Pick model based on redis or default setting
        model = hvalues.get('model_name', '')
        version = hvalues.get('model_version', '')
        if not model or not version:
            model, version = settings.TRACKING_MODEL.split(':')

        t = timeit.default_timer()
        model = TrackingClient(hostname,
                               redis_hash,
                               model,
                               int(version),
                               progress_callback=self._update_progress)

        self.logger.debug('Created the TrackingClient in %s seconds.',
                          timeit.default_timer() - t)
        return model

    def _get_tracker(self, redis_hash, hvalues, raw, segmented):
        tracking_model = self._get_model(redis_hash, hvalues)

        # Some tracking models do not have an ImageNormalization Layer.
        # If not, the data must be normalized before being tracked.
        if settings.NORMALIZE_TRACKING:
            for frame in range(raw.shape[0]):
                raw[frame, :, :, 0] = processing.normalize(raw[frame, :, :, 0])

        features = {'appearance', 'distance', 'neighborhood', 'regionprop'}
        tracker = tracking.CellTracker(raw, segmented,
                                       tracking_model,
                                       max_distance=settings.MAX_DISTANCE,
                                       track_length=settings.TRACK_LENGTH,
                                       division=settings.DIVISION,
                                       birth=settings.BIRTH,
                                       death=settings.DEATH,
                                       neighborhood_scale_size=settings.NEIGHBORHOOD_SCALE_SIZE,
                                       features=features)

        self.logger.debug('Created tracker!')
        return tracker

    def _update_progress(self, redis_hash, progress):
        self.update_key(redis_hash, {
            'progress': progress,
        })

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
        raw = utils.get_image(os.path.join(subdir, fname))

        # remove the last dimensions added by `get_image`
        tiff_stack = np.squeeze(raw, -1)  # TODO: required? check the ndim?
        if len(tiff_stack.shape) != 3:
            raise ValueError("This tiff file has shape {}, which is not 3 "
                             "dimensions. Tracking can only be done on images "
                             "with 3 dimensions, (time, width, height)".format(
                                 tiff_stack.shape))

        # Calculate scale of a subset of raw
        scale = hvalues.get('scale', '')
        if not scale:
            # Detect scale of image
            scale = self.detect_scale(tiff_stack)
            self.logger.debug('Image scale detected: %s', scale)
            self.update_key(redis_hash, {'scale': scale})
        else:
            scale = float(scale)
            self.logger.debug('Image scale already calculated: %s', scale)

        # Pick model and postprocess based on either label or defaults
        if settings.LABEL_DETECT_ENABLED:
            label = self.detect_label(tiff_stack)  # Predict label type

            # Get appropriate model and postprocess function for the label
            model_name, model_version = utils._pick_model(label)
            postprocess_function = utils._pick_postprocess(label)
        else:
            label = 99  # Equivalent to none
            model_name, model_version = settings.TRACKING_SEGMENT_MODEL.split(':')
            postprocess_function = settings.TRACKING_POSTPROCESS_FUNCTION

        num_frames = len(tiff_stack)
        hash_to_frame = {}
        remaining_hashes = set()

        self.logger.debug('Got tiffstack shape %s.', tiff_stack.shape)
        self.logger.debug('tiffstack num_frames %s.', num_frames)

        with utils.get_tempdir() as tempdir:
            for (i, img) in enumerate(tiff_stack):
                # make a file name for this frame
                segment_fname = '{}-tracking-frame-{}.tif'.format(
                    hvalues.get('original_name'), i)
                segment_local_path = os.path.join(tempdir, segment_fname)

                # upload it
                tifffile.imsave(segment_local_path, img)
                upload_file_name, upload_file_url = self.storage.upload(
                    segment_local_path)

                # prepare hvalues for this frame's hash
                current_timestamp = self.get_current_timestamp()
                frame_hvalues = {
                    'identity_upload': self.hostname,
                    'input_file_name': upload_file_name,
                    'original_name': segment_fname,
                    'model_name': model_name,
                    'model_version': model_version,
                    'postprocess_function': postprocess_function,
                    'cuts': settings.CUTS,
                    'status': 'new',
                    'created_at': current_timestamp,
                    'updated_at': current_timestamp,
                    'url': upload_file_url,
                    'scale': scale,
                    'label': str(label)
                }

                self.logger.debug("Setting %s", frame_hvalues)

                # make a hash for this frame
                segment_hash = '{prefix}:{file}:{hash}'.format(
                    prefix=settings.SEGMENTATION_QUEUE,
                    file=segment_fname,
                    hash=uuid.uuid4().hex)

                # push the hash to redis and the predict queue
                self.redis.hmset(segment_hash, frame_hvalues)
                self.redis.lpush(settings.SEGMENTATION_QUEUE, segment_hash)
                self.logger.debug('Added new hash for segmentation `%s`: %s',
                                  segment_hash, json.dumps(frame_hvalues,
                                                           indent=4))
                hash_to_frame[segment_hash] = i
                remaining_hashes.add(segment_hash)

            # pop hash, check it, and push it back if it's not done
            # this checks the same hash over and over again, since set's
            # pop is not random. This is fine, since we still need every
            # hash to finish before doing anything.
            frames = {}
            while remaining_hashes:
                finished_hashes = set()

                self.logger.debug('Checking on hashes.')
                for segment_hash in remaining_hashes:
                    status = self.redis.hget(segment_hash, 'status')

                    self.logger.debug('Hash %s has status %s',
                                      segment_hash, status)

                    if status == self.failed_status:
                        reason = self.redis.hget(segment_hash, 'reason')
                        raise RuntimeError(
                            'Tracking failed during segmentation on frame {}.'
                            '\nSegmentation Error: {}'.format(
                                hash_to_frame[segment_hash], reason))

                    elif status == self.final_status:
                        # if it's done, save the frame, as they'll be packed up
                        # later
                        frame_zip = self.storage.download(
                            self.redis.hget(segment_hash, 'output_file_name'),
                            tempdir)

                        frame_files = list(utils.iter_image_archive(frame_zip,
                                                                    tempdir))

                        if len(frame_files) != 1:
                            raise RuntimeError(
                                "After unzipping predicted frame, got "
                                "back multiple files {}. Expected a "
                                "single file.".format(frame_files))

                        frame_idx = hash_to_frame[segment_hash]
                        frames[frame_idx] = utils.get_image(frame_files[0])
                        finished_hashes.add(segment_hash)

                remaining_hashes -= finished_hashes
                time.sleep(settings.INTERVAL)

        frames = [frames[i] for i in range(num_frames)]

        # Cast y to int to avoid issues during fourier transform/drift correction
        return {"X": np.expand_dims(tiff_stack, axis=-1), "y": np.array(frames, dtype='uint16')}

    def _consume(self, redis_hash):
        hvalues = self.redis.hgetall(redis_hash)
        self.logger.debug('Found `%s:*` hash to process "%s": %s',
                          self.queue, redis_hash, json.dumps(hvalues, indent=4))

        if hvalues.get('status') in self.finished_statuses:
            self.logger.warning('Found completed hash `%s` with status %s.',
                                redis_hash, hvalues.get('status'))
            return hvalues.get('status')

        # Set status and initial progress
        self.update_key(redis_hash, {
            'status': 'started',
            'progress': 0,
        })

        with utils.get_tempdir() as tempdir:
            fname = self.storage.download(hvalues.get('input_file_name'),
                                          tempdir)
            data = self._load_data(redis_hash, tempdir, fname)

            self.logger.debug('Got contents tracking file contents.')
            self.logger.debug('X shape: %s', data['X'].shape)
            self.logger.debug('y shape: %s', data['y'].shape)

            # Correct for drift if enabled
            if settings.DRIFT_CORRECT_ENABLED:
                t = timeit.default_timer()
                data['X'], data['y'] = processing.correct_drift(data['X'], data['y'])
                self.logger.debug('Drift correction complete in %s seconds.',
                                  timeit.default_timer() - t)

            # TODO Add support for rescaling in the tracker
            tracker = self._get_tracker(redis_hash, hvalues,
                                        data['X'], data['y'])
            self.logger.debug('Trying to track...')

            tracker.track_cells()

            self.logger.debug('Tracking done!')

            # Post-process and save the output file
            tracked_data = tracker.postprocess()

            # Save lineage data to JSON file
            lineage_file = os.path.join(tempdir, 'lineage.json')
            with open(lineage_file, 'w') as fp:
                json.dump(tracked_data['tracks'], fp)

            save_name = hvalues.get('original_name', fname)
            subdir = os.path.dirname(save_name.replace(tempdir, ''))
            name = os.path.splitext(os.path.basename(save_name))[0]

            # Save tracked data as tiff stack
            outpaths = utils.save_numpy_array(
                tracked_data['y_tracked'], name=name,
                subdir=subdir, output_dir=tempdir)

            outpaths.append(lineage_file)

            # Save as zip instead of .trk for demo-purposes
            zip_file = utils.zip_files(outpaths, tempdir)

            output_file_name, output_url = self.storage.upload(zip_file)

            self.update_key(redis_hash, {
                'status': self.final_status,
                'output_url': output_url,
                'output_file_name': output_file_name,
                'finished_at': self.get_current_timestamp(),
            })
        return self.final_status
