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
"""Utility functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import time
import timeit
import hashlib
import logging
import tarfile
import zipfile

import numpy as np
import PIL
import tifffile
from tensorflow.keras.preprocessing.image import img_to_array


logger = logging.getLogger('redis_consumer.utils')


def strip_bucket_path(path):
    """Remove leading/trailing '/'s from cloud bucket folder names"""
    return '/'.join(y for y in path.split('/') if y)


def iter_image_archive(zip_path, destination):
    """Extract all files in archive and yield the paths of all images.

    Args:
        zip_path: path to zip archive
        destination: path to extract all images

    Returns:
        Iterator of all image paths in extracted archive
    """
    archive = zipfile.ZipFile(zip_path, 'r', allowZip64=True)
    is_valid = lambda x: os.path.splitext(x)[1] and '__MACOSX' not in x
    for info in archive.infolist():
        extracted = archive.extract(info, path=destination)
        if os.path.isfile(extracted):
            if is_valid(extracted):
                yield extracted


def get_image_files_from_dir(fname, destination=None):
    """Based on the file, returns a list of all images in that file.

    Args:
        fname: file (image or zip file)
        destination: folder to save image files from archive, if applicable

    Returns:
        list of image file paths
    """
    if zipfile.is_zipfile(fname):
        archive = iter_image_archive(fname, destination)
        for f in archive:
            yield f
    else:
        yield fname


def get_image(filepath):
    """Open image file as numpy array.

    Args:
        filepath: full filepath of image file

    Returns:
        img: numpy array of image data
    """
    logger.debug('Loading %s into numpy array', filepath)
    if os.path.splitext(filepath)[-1].lower() in {'.tif', '.tiff'}:
        img = tifffile.TiffFile(filepath).asarray()
        # tiff files should not have a channel dim
        # img = np.expand_dims(img, axis=-1)
    else:
        img = img_to_array(PIL.Image.open(filepath))

    logger.debug('Loaded %s into numpy array with shape %s',
                 filepath, img.shape)
    return img.astype('float32')


def save_numpy_array(arr, name='', subdir='', output_dir=None):
    """Split tensor into channels and save each as a tiff file.

    Args:
        arr: numpy array of image data
        name: name of original input image file
        subdir: optional subdirectory to save the result.
        output_dir: base directory for features

    Returns:
        out_paths: list of all saved image paths
    """
    logger.debug('Saving array of size %s', arr.shape)

    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, -1)
        logger.debug('Expanding dimension of array to include channel '
                     'dimension. New shape is %s', arr.shape)

    start = timeit.default_timer()
    output_dir = output_dir if output_dir is None else output_dir
    if subdir.startswith(os.path.sep):
        subdir = subdir[1:]

    out_paths = []
    for channel in range(arr.shape[-1]):
        try:
            logger.debug('Saving channel %s', channel)
            img = arr[..., channel].astype('float32')

            _name = 'feature_{}.tif'.format(channel)
            if name:
                _name = '{}_{}'.format(name, _name)

            path = os.path.join(output_dir, subdir, _name)

            # Create subdirs if they do not exist
            if not os.path.isdir(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            tifffile.imsave(path, img)
            logger.debug('Saved channel %s to %s', channel, path)
            out_paths.append(path)
        except Exception as err:  # pylint: disable=broad-except
            out_paths = []
            logger.error('Could not save channel %s as image: %s',
                         channel, err)
    logger.debug('Saved %s image files in %s seconds.',
                 len(out_paths), timeit.default_timer() - start)
    return out_paths


def load_track_file(filename):
    """Load a trk/trks file.
    Args:
        trks_file: full path to the file including .trk/.trks
    Returns:
        A dictionary with raw, tracked, and lineage data
    """
    if filename.endswith(".trk") or filename.endswith(".trks"):
        with tarfile.open(filename, 'r') as trks:

            # numpy can't read these from disk...
            array_file = io.BytesIO()
            array_file.write(trks.extractfile('raw.npy').read())
            array_file.seek(0)
            raw = np.load(array_file)
            array_file.close()

            array_file = io.BytesIO()
            array_file.write(trks.extractfile('tracked.npy').read())
            array_file.seek(0)
            tracked = np.load(array_file)
            array_file.close()

        return {'X': raw, 'y': tracked}

    raise Exception("track file must end with .zip or .trk/.trks")


def zip_files(files, dest=None, prefix=None):
    """Save files in zip archive and return the path.

    Args:
        files: all filepaths that will be saved in the zip
        dest: saves zip file to this directory, OUTPUT_DIR by default

    Returns:
        zip_filename: filepath to new zip archive
    """
    start = timeit.default_timer()
    filename = '{prefix}{join}{hash}.zip'.format(
        prefix=prefix if prefix else '',
        join='_' if prefix else '',
        hash=hashlib.md5(str(time.time()).encode('utf-8')).hexdigest())

    filepath = os.path.join(dest, filename)

    zip_kwargs = {
        'mode': 'w',
        'compression': zipfile.ZIP_DEFLATED,
        'allowZip64': True,
    }

    try:
        logger.debug('Saving %s files to %s.', len(files), filepath)
        with zipfile.ZipFile(filepath, **zip_kwargs) as zf:
            for f in files:  # writing each file one by one
                name = f.replace(dest, '')
                name = name[1:] if name.startswith(os.path.sep) else name
                zf.write(f, arcname=name)
        logger.debug('Saved %s files to %s in %s seconds.',
                     len(files), filepath, timeit.default_timer() - start)
    except Exception as err:
        logger.error('Failed to write zipfile: %s', err)
        raise err
    return filepath
