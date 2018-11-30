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
"""Classes to consume events in redis"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import tempfile
import zipfile
from hashlib import md5
from time import sleep, time
from timeit import default_timer

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.external import tifffile as tiff
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import watershed
from skimage.morphology import remove_small_objects, dilation, erosion
from keras_preprocessing.image import img_to_array
from tornado import ioloop

from redis_consumer.settings import OUTPUT_DIR


class Consumer(object):
    """Base class for all redis event consumer classes"""

    def __init__(self,
                 redis_client,
                 storage_client,
                 final_status='done'):
        self.output_dir = OUTPUT_DIR
        self.redis = redis_client
        self.storage = storage_client
        self.final_status = final_status
        self.logger = logging.getLogger(str(self.__class__.__name__))

    def iter_redis_hashes(self, status='new', prefix='predict'):
        """Iterate over hash values in redis
        and yield each with the given status value.
        # Returns: Iterator of all hashes with a valid status
        """
        for key in self.redis.keys():
            # Check if the key is a hash
            if self.redis.type(key) == 'hash':
                # Check if necessary to filter based on prefix
                if prefix is not None:
                    # Wrong prefix, skip it.
                    if not key.startswith(str(prefix).lower()):
                        continue

                # if status is given, only yield hashes with that status
                if status is not None:
                    if self.redis.hget(key, 'status') == str(status):
                        yield key
                else:  # no need to check the status
                    yield key

    def _handle_error(self, err, redis_hash):
        # Update redis with failed status
        self.redis.hmset(redis_hash, {
            'reason': err,
            'status': 'failed'
        })
        self.logger.error('Failed to process redis key %s. Error: %s',
                          redis_hash, err)


    def is_zip_file(self, filename):
        """Returns boolean if cloud file is a zip file
        If using on local file, use ZipFile.is_zipfile instead
        # Arguments:
            filename: key of file in cloud storage
        # Returns:
            True if file is a zip archive otherwise False
        """
        return os.path.splitext(filename)[-1].lower() == '.zip'

    def get_image(self, filepath):
        """Open image file as numpy array
        # Arguments:
            filepath: full filepath of image file
        # Returns:
            img: numpy array of image data
        """
        self.logger.debug('Loading %s into numpy array', filepath)
        if os.path.splitext(filepath)[-1].lower() in {'.tif', '.tiff'}:
            img = np.float32(tiff.TiffFile(filepath).asarray())
            # check for channel axis
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
        else:
            img = img_to_array(Image.open(filepath))

        self.logger.debug('Loaded image into numpy array with '
                          'shape %s', img.shape)
        return img

    def save_zip_file(self, files, dest=None):
        """Save files in zip archive and return the path
        # Arguments:
            files: all filepaths that will be saved in the zip
            dest: saves zip file to this directory, OUTPUT_DIR by default
        # Returns:
            zip_filename: filepath to new zip archive
        """
        try:
            output_dir = self.output_dir if dest is None else dest
            filename = 'prediction_{}'.format(time()).encode('utf-8')
            hashed_filename = '{}.zip'.format(md5(filename).hexdigest())
            zip_filename = os.path.join(output_dir, hashed_filename)
            self.logger.warning(zip_filename)
            # Create ZipFile and Write each file to it
            with zipfile.ZipFile(zip_filename, 'w') as zip_file:
                for f in files:  # writing each file one by one
                    self.logger.warning(f)
                    name = f.replace(output_dir, '')
                    if name.startswith(os.path.sep):
                        name = name[1:]
                    self.logger.warning(name)
                    zip_file.write(f, arcname=name)
            return zip_filename
        except Exception as err:
            self.logger.error('Failed to write zipfile: %s', err)
            raise err

    def iter_image_archive(self, zip_path, destination):
        """Extract all files in archie and yield the paths of all images
        # Arguments:
            zip_path: path to zip archive
            destination: path to extract all images
        # Returns:
            Iterator of all image paths in extracted archive
        """
        archive = zipfile.ZipFile(zip_path, 'r')
        is_valid = lambda x: os.path.splitext(x)[1] and '__MACOSX' not in x
        for info in archive.infolist():
            try:
                extracted = archive.extract(info, path=destination)
                if os.path.isfile(extracted):
                    if is_valid(extracted):
                        yield extracted
            except:
                self.logger.warning('Could not extract %s', info.filename)

    def save_numpy_array(self, arr, name='', subdir='', output_dir=None):
        """Split tensor into channels and save each as a tiff
        # Arguments:
            arr: numpy array of image data
            name: name of original input image file
            subdir: optional subdirectory to save the result.
            output_dir: base directory for features
        # Returns:
            out_paths: list of all saved image paths
        """
        output_dir = self.output_dir if output_dir is None else output_dir
        if subdir.startswith(os.path.sep):
            subdir = subdir[1:]

        out_paths = []
        for channel in range(arr.shape[-1]):
            try:
                self.logger.debug('saving channel %s', channel)
                img = arr[..., channel].astype('float32')

                _name = 'feature_{}.tif'.format(channel)
                if name:
                    _name = '{}_{}'.format(name, _name)

                path = os.path.join(output_dir, subdir, _name)

                # Create subdirs if they do not exist
                if not os.path.isdir(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))

                tiff.imsave(path, img)
                self.logger.debug('saved channel %s to %s', channel, path)
                out_paths.append(path)
            except Exception as err:
                out_paths = []
                self.logger.error('Could not save channel %s as image: %s',
                                  channel, err)
        return out_paths

    def _consume(self, redis_hash):
        raise NotImplementedError

    def consume(self, interval, status=None, prefix=None):
        """Consume all redis events every `interval` seconds
        # Arguments:
            interval: waits this many seconds between consume calls
        # Returns:
            nothing: this is the consumer main process
        """
        if not str(interval).isdigit():
            raise ValueError('Expected `interval` to be a number. '
                             'Got {}'.format(type(interval)))

        while True:
            try:
                # process each unprocessed hash
                for redis_hash in self.iter_redis_hashes(status, prefix):
                    start = default_timer()
                    self._consume(redis_hash)
                    self.logger.debug('Consumed key %s in %s s',
                                      redis_hash, default_timer() - start)
            except Exception as err:
                self.logger.error(err)

            sleep(interval)


class PredictionConsumer(Consumer):
    """Consumer to send image data to tf-serving and upload the results"""

    def __init__(self,
                 redis_client,
                 storage_client,
                 tf_client,
                 final_status='done'):
        self.tf_client = tf_client
        super(PredictionConsumer, self).__init__(
            redis_client, storage_client, final_status)

    def pad_image(self, image, field):
        """Pad each the input image for proper dimensions when stitiching
        # Arguments:
            image: np.array of image data
            field: receptive field size of model
        # Returns:
            image data padded in the x and y axes
        """
        window = (field - 1) // 2
        # Pad images by the field size in the x and y axes
        pad_width = []
        for i in range(len(image.shape)):
            if i == image.ndim - 3:
                pad_width.append((window, window))
            elif i == image.ndim - 2:
                pad_width.append((window, window))
            else:
                pad_width.append((0, 0))

        return np.pad(image, pad_width, mode='reflect')

    def _iter_cuts(self, img, cuts, field):
        crop_x = img.shape[img.ndim - 3] // cuts
        crop_y = img.shape[img.ndim - 2] // cuts
        for i in range(cuts):
            for j in range(cuts):
                a, b = i * crop_x, (i + 1) * crop_x
                c, d = j * crop_y, (j + 1) * crop_y
                yield a, b, c, d

    def process_big_image(self, cuts, img, field, model_name, model_version):
        """Slice big image into smaller images for prediction,
        then stitches all the smaller images back together
        # Arguments:
            cuts: number of cuts in x and y to slice smaller images
            img: image data as numpy array
            field: receptive field size of model, changes padding sizes
            model_name: hosted model to send image data
            model_version: model version to query
        # Returns:
            tf_results: single numpy array of predictions on big input image
        """
        cuts = int(cuts)
        win_x, win_y = (field - 1) // 2, (field - 1) // 2

        tf_results = None  # Channel shape is unknown until first request
        padded_img = self.pad_image(img, field)

        images, coords = [], []

        for a, b, c, d in self._iter_cuts(img, cuts, field):
            if img.ndim >= 4:
                data = padded_img[:, a:b + 2 * win_x, c:d + 2 * win_y, :]
            else:
                data = padded_img[a:b + 2 * win_x, c:d + 2 * win_y, :]

            images.append(data)
            coords.append((a, b, c, d))

        def post_many():
            timeout = 300 * len(images)
            clients = len(images)
            return self.tf_client.tornado_images(
                images, model_name, model_version,
                timeout=timeout, max_clients=clients)

        predicted = ioloop.IOLoop.current().run_sync(post_many)

        for (a, b, c, d), pred in zip(coords, predicted):
            if tf_results is None:
                tf_results = np.zeros(list(img.shape)[:-1] + [pred.shape[-1]])
                self.logger.debug('initialized output tensor of shape %s',
                                  tf_results.shape)

            if pred.ndim >= 4:
                tf_results[:, a:b, c:d, :] = pred[:, win_x:-win_x, win_y:-win_y, :]
            else:
                tf_results[a:b, c:d, :] = pred[win_x:-win_x, win_y:-win_y, :]

        return tf_results

    def process_image(self, filename, model_name, model_version, cuts=0, field=61):
        """POSTs image data to tf-serving then saves the result
        as a zip and uploads into the cloud bucket.
        # Arguments:
            filename: path to cloud destination of image file
            model_name: name of model in tf-serving
            model_version: integer version number of model in tf-serving
            cuts: if > 1, slices large images and predicts on each slice
            field: receptive field of model
        # Returns:
            output_url: URL of results zip file
        """
        with tempfile.TemporaryDirectory() as tempdir:
            local_fname = self.storage.download(filename, tempdir)
            img = self.get_image(local_fname)

            if int(cuts) > 1:
                tf_results = self.process_big_image(
                    cuts, img, field, model_name, model_version)
            else:
                # Get tf-serving predictions of image
                tf_results = self.tf_client.post_image(
                    img, model_name, model_version)

            # Save each tf-serving prediction channel as image file
            out_paths = self.save_numpy_array(
                tf_results, name=local_fname, output_dir=tempdir)

            # Save each prediction image as zip file
            zip_file = self.save_zip_file(out_paths, tempdir)

            # Upload the zip file to cloud storage bucket
            output_url = self.storage.upload(zip_file)
            self.logger.debug('Saved output to: "%s"', output_url)

        return output_url

    def process_zip(self, filename, model_name, model_version, cuts=0, field=61):
        """Process each image inside a zip archive and save/upload resulting
        zip archive with mirrored folder structure.
        # Arguments:
            filename: path to cloud destination of image file
            model_name: name of model in tf-serving
            model_version: integer version number of model in tf-serving
            cuts: if > 1, slices large images and predicts on each slice
            field: receptive field of model
        # Returns:
            output_url: URL of results zip file
        """
        with tempfile.TemporaryDirectory() as tempdir:
            local_fname = self.storage.download(filename, tempdir)
            if not zipfile.is_zipfile(local_fname):
                self.logger.error('Invalid zip file: %s', local_fname)
                raise ValueError('{} is not a zipfile'.format(local_fname))

            image_files = [f for f in self.iter_image_archive(local_fname, tempdir)]
            images = (self.get_image(f) for f in image_files)

            def post_many():
                timeout = 300 * len(image_files)
                clients = len(image_files)
                return self.tf_client.tornado_images(
                    images, model_name, model_version,
                    timeout=timeout, max_clients=clients)

            tf_results = ioloop.IOLoop.current().run_sync(post_many)

            all_output = []
            # Save each tf-serving prediction channel as image file
            for results, imfile in zip(tf_results, image_files):
                subdir = os.path.dirname(imfile.replace(tempdir, ''))
                name = os.path.splitext(os.path.basename(imfile))[0]

                _out_paths = self.save_numpy_array(
                    results, name=name, subdir=subdir, output_dir=tempdir)

                all_output.extend(_out_paths)

            # Save each prediction image as zip file
            zip_file = self.save_zip_file(all_output, tempdir)

            # Upload the zip file to cloud storage bucket
            upload_dest = self.storage.upload(zip_file)

        return upload_dest

    def _consume(self, redis_hash):
        hash_values = self.redis.hgetall(redis_hash)
        self.logger.debug('Found hash to process "%s": %s',
                            redis_hash, json.dumps(hash_values, indent=4))

        self.redis.hset(redis_hash, 'status', 'processing')
        self.logger.debug('processing image: %s', redis_hash)

        try:
            start = timeit.default_timer()
            fname = hash_values.get('file_name')

            if self.is_zip_file(fname):
                _func = self.process_zip
            else:
                _func = self.process_image

            uploaded_file_path = _func(
                fname,
                hash_values.get('model_name'),
                hash_values.get('model_version'),
                hash_values.get('cuts'))

            # output_url = self.storage.get_public_url(uploaded_file_path)
            # self.logger.debug('Saved output to: "%s"', output_url)

            self.logger.debug('Processed key %s in %s s',
                                redis_hash, timeit.default_timer() - start)

            # Update redis with the results
            self.redis.hmset(redis_hash, {
                'file_name': uploaded_file_path,
                'status': self.final_status
            })

        except Exception as err:
            self._handle_error(err, redis_hash)

    def consume(self, interval, status='new', prefix='predict'):
        # verify that tf-serving is ready to accept images
        self.tf_client.verify_endpoint_liveness()
        super(PredictionConsumer, self).consume(interval)


class ProcessingConsumer(Consumer):
    """Base class for pre- and post-processing Consumers"""

    def __init__(self,
                 redis_client,
                 storage_client,
                 hash_prefix='predict',
                 watch_status='new',
                 final_status='done'):
        self._processing_dict = {}
        super(ProcessingConsumer, self).__init__(
            redis_client, storage_client,
            hash_prefix, watch_status, final_status)

    def _process_data(self, data, fnkey):
        """Get the post-process function based on the `function_key` input
        and apply it to the image if it is not None.
        # Arguments:
            data: numpy data to process
            fnkey: string key to lookup the transformation function
        # Returns:
            processed_data: numpy data processed by the transformation function
        """
        processing_function = self._processing_dict.get(fnkey)
        if processing_function is not None:
            processed_data = processing_function(data)
        else:
            processed_data = data
        return processed_data

    def process_image(self, filename, fnkey=None):
        """Processes a single image file and re-uploads the processed image as
        `${original_name}_${fnkey}.tif`
        # Arguments:
            filename: key of file in cloud storage
            fnkey: processing function key
        # Returns:
            processed_filename: new filepath for processed image
        """
        # If no processing function is given, do nothing
        if fnkey is None:
            return filename

        processed_filename = '{}_{}.tif'.format(
            os.path.splitext(filename)[0], fnkey)

        with tempfile.TemporaryDirectory() as tempdir:
            local_fname = self.storage.download(filename, download_dir=tempdir)
            processed_filename = os.path.join(tempdir, processed_filename)

            img = self.get_image(local_fname)
            processed_img = self._process_data(img, fnkey)

            # Save preprocessed data as image file
            tiff.imsave(processed_filename, processed_img)

            # Upload the zip file to cloud storage bucket
            uploaded_dest = self.storage.upload(processed_filename)

        return uploaded_dest

    def process_zip(self, filename, fnkey):
        """Processed all image files in the archive and re-uploads the
        processed archive as `${original_name}_${fnkey}.zip`
        # Arguments:
            filename: key of file in cloud storage
            fnkey: processing function key
        # Returns:
            processed_filename: new filepath for processed zip archive
        """
        # If no preprocessing function is given, do nothing
        if fnkey is None:
            return filename

        with tempfile.TemporaryDirectory() as tempdir:
            local_fname = self.storage.download(filename, download_dir=tempdir)

            if not zipfile.is_zipfile(local_fname):
                self.logger.error('Invalid zip file: %s', local_fname)
                raise ValueError('{} is not a zipfile'.format(local_fname))

            all_output = []
            for imfile in self.iter_image_archive(local_fname, tempdir):
                image = self.get_image(imfile)
                processed_image = self._process_data(image, fnkey)

                # Save each tf-serving prediction channel as image file
                subdir = os.path.dirname(imfile.replace(tempdir, ''))
                name = os.path.splitext(os.path.basename(imfile))[0]
                _out_paths = self.save_numpy_array(
                    processed_image, name=name, subdir=subdir,
                    output_dir=tempdir)

                all_output.extend(_out_paths)

            # Save each prediction image as zip file
            zip_file = self.save_zip_file(all_output, tempdir)

            # Upload the zip file to cloud storage bucket
            upload_dest = self.storage.upload(zip_file)

        return upload_dest

    def process(self, filename, fnkey):
        """Determine if image or zip and call the appropriate function
        # Arguments:
            filename: key of file in cloud storage
            fnkey: preprocessing function key
        # Returns:
            processed_filename: new filepath for pre-processed file
        """
        # If no postprocessing function is given, do nothing
        if fnkey is None:
            uploaded_file_path = filename
        elif self.is_zip_file(filename):
            uploaded_file_path = self.process_zip(filename, fnkey)
        else:
            uploaded_file_path = self.process_image(filename, fnkey)
        return uploaded_file_path


class PreProcessingConsumer(ProcessingConsumer):
    """Preprocess each image in redis before Prediction"""

    def __init__(self,
                 redis_client,
                 storage_client,
                 hash_prefix='predict',
                 watch_status='new',
                 final_status='preprocessed'):
        super(PreProcessingConsumer, self).__init__(
            redis_client, storage_client,
            hash_prefix, watch_status, final_status)
        # TODO: Add more preprocessing functions here
        self._processing_dict = {
            'normalize': self.normalize_image
        }

    def normalize_image(self, image):
        """Normalize image data by dividing by the maximum pixel value
        # Arguments:
            image: numpy array of image data
        # Returns:
            normal_image: normalized image data
        """
        normal_image = image * 255.0 / image.max()
        return normal_image

    def _consume(self, redis_hash):
        hash_values = self.redis.hgetall(redis_hash)
        self.logger.debug('Found hash to preprocess "%s": %s',
                            redis_hash, json.dumps(hash_values, indent=4))

        self.redis.hset(redis_hash, 'status', 'preprocessing')
        self.logger.debug('Pre-processing image: %s', redis_hash)

        try:
            start = timeit.default_timer()
            uploaded_file_path = self.process(
                hash_values.get('file_name'),
                hash_values.get('preprocess_function'))

            self.redis.hmset(redis_hash, {
                'status': self.final_status,
                'file_name': uploaded_file_path
            })

            self.logger.debug('Pre-processed key %s in %s s',
                                redis_hash, timeit.default_timer() - start)

        except Exception as err:
            self._handle_error(err, redis_hash)


class PostProcessingConsumer(ProcessingConsumer):
    """Post Process each prediction"""

    def __init__(self,
                 redis_client,
                 storage_client,
                 hash_prefix='predict',
                 watch_status='processed',
                 final_status='done'):
        super(PostProcessingConsumer, self).__init__(
            redis_client, storage_client,
            hash_prefix, watch_status, final_status)
        # TODO: Add more postprocessing functions here
        self._processing_dict = {
            'watershed': self.watershed,
            'deepcell': self.deepcell,
            'mibi': self.mibi
        }

    def watershed(self, image, min_distance=10, threshold_abs=0.05):
        """Use the watershed method to identify unique cells based
        on their distance transform.
        # TODO: labels should be the fgbg output, NOT the union of distances
        # TODO: as is, there are small patches of pixels that are garbage
        # Arguments:
            image: distance transform of image (model output)
            min_distance: minimum number of pixels separating peaks
            threshold_abs: minimum intensity of peaks
        # Returns:
            image mask where each cell is annotated uniquely
        """
        self.logger.debug('performing watershed segmentation postprocessing '
                          'on image with shape %s', image.shape)

        distance = np.argmax(image, axis=-1)
        labels = (distance > 0).astype('int')

        local_maxi = peak_local_max(
            image[..., -1],
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            indices=False,
            labels=labels,
            exclude_border=False)

        markers = ndimage.label(local_maxi)[0]
        segments = watershed(-distance, markers, mask=labels)
        results = np.expand_dims(segments, axis=-1)
        results = remove_small_objects(
            results, min_size=50, connectivity=1)
        return results

    def deepcell(self, image, threshold=.8):
        interior = image[..., 2] > threshold
        data = np.expand_dims(interior, axis=-1)
        labeled = ndimage.label(data)[0]
        labeled = remove_small_objects(
            labeled, min_size=50, connectivity=1)
        return labeled

    def mibi(self, predictions, threshold=.25):
        edge_thresh = threshold
        interior_thresh = threshold

        def dilate(array, mask, num_dilations):
            copy = np.copy(array)
            for _ in range(0, num_dilations):
                dilated = dilation(copy)
                # if still within the mask range AND one cell not eating another, dilate
                copy = np.where((mask != 0) & (dilated != copy) & (copy == 0), dilated, copy)
            return copy
        def dilate_nomask(array, num_dilations):
            copy = np.copy(array)
            for _ in range(0, num_dilations):
                dilated = dilation(copy)
                # if one cell not eating another, dilate
                copy = np.where((dilated != copy) & (copy == 0), dilated, copy)
            return copy

        def erode(array, num_erosions):
            original = np.copy(array)
            for _ in range(0, num_erosions):
                eroded = erosion(np.copy(original))
                original[original != eroded] = 0
            return original

        edge = np.copy(predictions[..., 0])
        edge[edge < edge_thresh] = 0
        edge[edge >= edge_thresh] = 1

        interior = np.copy(predictions[..., 1])
        interior[interior >= interior_thresh] = 1
        interior[interior < interior_thresh] = 0

        # define foreground as the interior bounded by edge
        fg_thresh = np.logical_and(interior == 1, edge == 0)

        # remove small objects from the foreground segmentation
        fg_thresh = remove_small_objects(
            fg_thresh, min_size=50, connectivity=1)

        fg_thresh = np.expand_dims(fg_thresh, axis=-1)

        watershed_segmentation = label(np.squeeze(fg_thresh), connectivity=2)

        for _ in range(8):
            watershed_segmentation = dilate(watershed_segmentation, interior, 2)
            watershed_segmentation = erode(watershed_segmentation, 1)

        watershed_segmentation = dilate(watershed_segmentation, interior, 2)

        for _ in range(2):
            watershed_segmentation = dilate_nomask(watershed_segmentation, 1)
            watershed_segmentation = erode(watershed_segmentation, 2)

        watershed_segmentation = np.expand_dims(watershed_segmentation, axis=-1)
        return watershed_segmentation.astype('uint16')

    def process_image(self, filename, fnkey=None):
        """All predictions are zip images, so remove this function"""
        raise NotImplementedError('PostProcessors only process '
                                  'zip archives from tf-serving')

    def process_zip(self, filename, fnkey):
        """Processed all image files in the archive and re-uploads the
        processed archive as `${original_name}_${fnkey}.zip`
        # Arguments:
            filename: key of file in cloud storage
            fnkey: processing function key
        # Returns:
            processed_filename: new filepath for processed zip archive
        """
        # If no preprocessing function is given, do nothing
        if fnkey is None:
            return filename

        with tempfile.TemporaryDirectory() as tempdir:
            local_fname = self.storage.download(filename, download_dir=tempdir)

            if not zipfile.is_zipfile(local_fname):
                self.logger.error('Invalid zip file: %s', local_fname)
                raise ValueError('{} is not a zipfile'.format(local_fname))

            all_images = {}
            for imfile in self.iter_image_archive(local_fname, tempdir):
                # load all features of each image as a single numpy array
                base_imfile = 'feature_'.join(imfile.split('feature_')[:-1])
                if base_imfile.endswith('_'):
                    base_imfile = base_imfile[:-1]
                base_imfile = '{}_{}'.format(base_imfile, fnkey)

                if base_imfile not in all_images:
                    all_images[base_imfile] = []
                all_images[base_imfile].append(imfile)

            all_output = []
            for base, feature_files in all_images.items():
                num_channels = len(feature_files)
                image = None  # set as numpy array after loading first image
                for i, f in enumerate(sorted(feature_files)):
                    feat = self.get_image(f)
                    if image is None:
                        shape = tuple(list(feat.shape)[:-1] + [num_channels])
                        image = np.zeros(shape)

                    image[..., i] = feat[..., 0]

                processed_image = self._process_data(image, fnkey)

                # Save each tf-serving prediction channel as image file
                subdir = os.path.dirname(base.replace(tempdir, ''))
                name = os.path.splitext(os.path.basename(base))[0]
                _out_paths = self.save_numpy_array(
                    processed_image, name=name, subdir=subdir,
                    output_dir=tempdir)

                all_output.extend(_out_paths)

            # Save each prediction image as zip file
            zip_file = self.save_zip_file(all_output, tempdir)

            # Upload the zip file to cloud storage bucket
            upload_dest = self.storage.upload(zip_file)

        return upload_dest

    def _consume(self, redis_hash):
        hash_values = self.redis.hgetall(redis_hash)
        self.logger.debug('Found hash to preprocess "%s": %s',
                          redis_hash, json.dumps(hash_values, indent=4))

        self.redis.hset(redis_hash, 'status', 'postprocessing')
        self.logger.debug('Post-processing image: %s', redis_hash)

        try:
            start = timeit.default_timer()

            uploaded_file_path = self.process(
                hash_values.get('file_name'),
                hash_values.get('postprocess_function'))

            self.logger.debug('Post-processed key %s in %s s',
                                redis_hash, timeit.default_timer() - start)

            output_url = self.storage.get_public_url(uploaded_file_path)
            self.logger.debug('Saved output to: "%s"', output_url)

            # Update redis with the results
            self.redis.hmset(redis_hash, {
                'output_url': output_url,
                'status': self.final_status
            })

        except Exception as err:
            self._handle_error(err, redis_hash)
        self.tf_client.verify_endpoint_liveness(expected_code=404)
        self.dp_client.verify_endpoint_liveness(expected_code=200)
        super(PredictionConsumer, self).consume(interval, status, prefix)
