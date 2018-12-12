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
"""Utility functions for manipulating images"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import hashlib
import logging
import zipfile

import numpy as np
from PIL import Image
from skimage.external import tifffile as tiff
from keras_preprocessing.image import img_to_array


logger = logging.getLogger('redis_consumer.utils')


def get_image(filepath):
    """Open image file as numpy array
    # Arguments:
        filepath: full filepath of image file
    # Returns:
        img: numpy array of image data
    """
    logger.debug('Loading %s into numpy array', filepath)
    if os.path.splitext(filepath)[-1].lower() in {'.tif', '.tiff'}:
        img = np.float32(tiff.TiffFile(filepath).asarray())
        # tiff files should not have a channel dim
        img = np.expand_dims(img, axis=-1)
    else:
        img = img_to_array(Image.open(filepath))

    logger.debug('Loaded %s into numpy array with shape %s',
                 filepath, img.shape)
    return img


def pad_image(image, field):
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


def save_numpy_array(arr, name='', subdir='', output_dir=None):
    """Split tensor into channels and save each as a tiff
    # Arguments:
        arr: numpy array of image data
        name: name of original input image file
        subdir: optional subdirectory to save the result.
        output_dir: base directory for features
    # Returns:
        out_paths: list of all saved image paths
    """
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

            tiff.imsave(path, img)
            logger.debug('Saved channel %s to %s', channel, path)
            out_paths.append(path)
        except Exception as err:  # pylint: disable=broad-except
            out_paths = []
            logger.error('Could not save channel %s as image: %s',
                         channel, err)
    return out_paths


def zip_files(files, dest=None, prefix=None):
    """Save files in zip archive and return the path
    # Arguments:
        files: all filepaths that will be saved in the zip
        dest: saves zip file to this directory, OUTPUT_DIR by default
    # Returns:
        zip_filename: filepath to new zip archive
    """
    filename = '{prefix}{join}{hash}.zip'.format(
        prefix=prefix if prefix else '',
        join='_' if prefix else '',
        hash=hashlib.md5(str(time.time()).encode('utf-8')).hexdigest())

    filepath = os.path.join(dest, filename)

    try:
        logger.debug('Saving %s files to %s', len(files), filepath)
        with zipfile.ZipFile(filepath, 'w') as zip_file:
            for f in files:  # writing each file one by one
                name = f.replace(dest, '')
                name = name[1:] if name.startswith(os.path.sep) else name
                zip_file.write(f, arcname=name)
        logger.debug('Saved %s files to %s', len(files), filepath)
    except Exception as err:
        logger.error('Failed to write zipfile: %s', err)
        raise err
    return filepath