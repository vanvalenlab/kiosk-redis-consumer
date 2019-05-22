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
"""Utility functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import time
import timeit
import contextlib
import hashlib
import json
import logging
import shutil
import tarfile
import tempfile
import zipfile
import six

import numpy as np
import keras_preprocessing.image
import skimage
import dict_to_protobuf
import PIL

from redis_consumer.pbs.types_pb2 import DESCRIPTOR
from redis_consumer.pbs.tensor_pb2 import TensorProto
from redis_consumer.pbs.tensor_shape_pb2 import TensorShapeProto


logger = logging.getLogger('redis_consumer.utils')


dtype_to_number = {
    i.name: i.number for i in DESCRIPTOR.enum_types_by_name['DataType'].values
}

# TODO: build this dynamically
number_to_dtype_value = {
    1: 'float_val',
    2: 'double_val',
    3: 'int_val',
    4: 'int_val',
    5: 'int_val',
    6: 'int_val',
    7: 'string_val',
    8: 'scomplex_val',
    9: 'int64_val',
    10: 'bool_val',
    18: 'dcomplex_val',
    19: 'half_val',
    20: 'resource_handle_val'
}


def grpc_response_to_dict(grpc_response):
    # TODO: 'unicode' object has no attribute 'ListFields'
    # response_dict = dict_to_protobuf.protobuf_to_dict(grpc_response)
    # return response_dict
    grpc_response_dict = dict()

    for k in grpc_response.outputs:
        shape = [x.size for x in grpc_response.outputs[k].tensor_shape.dim]

        logger.debug('Key: %s, shape: %s', k, shape)

        dtype_constant = grpc_response.outputs[k].dtype

        if dtype_constant not in number_to_dtype_value:
            grpc_response_dict[k] = 'value not found'
            logger.error('Tensor output data type not supported. '
                         'Returning empty dict.')

        dt = number_to_dtype_value[dtype_constant]
        if shape == [1]:
            grpc_response_dict[k] = eval(
                'grpc_response.outputs[k].' + dt)[0]
        else:
            grpc_response_dict[k] = np.array(
                eval('grpc_response.outputs[k].' + dt)).reshape(shape)

    return grpc_response_dict


def make_tensor_proto(data, dtype, shape=None):
    if shape is None:
        tensor_proto = TensorProto()
    else:
        tensor_shape = TensorShapeProto(dim=[TensorShapeProto.Dim(size=dim)
                                        for dim in shape])
        tensor_proto = TensorProto(tensor_shape=tensor_shape)

    if isinstance(dtype, six.string_types):
        dtype = dtype_to_number[dtype]

    dim = [{'size': 1}]
    values = [data]

    if hasattr(data, 'shape'):
        dim = [{'size': dim} for dim in data.shape]
        values = list(data.reshape(-1))

    tensor_proto_dict = {
        'dtype': dtype,
        'tensor_shape': {
            'dim': dim
        },
        number_to_dtype_value[dtype]: values
    }

    dict_to_protobuf.dict_to_protobuf(tensor_proto_dict, tensor_proto)

    return tensor_proto


# Workaround for python2 not supporting `with tempfile.TemporaryDirectory() as`
# These are unnecessary if not supporting python2
@contextlib.contextmanager
def cd(newdir, cleanup=lambda: True):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
        cleanup()


@contextlib.contextmanager
def get_tempdir():
    dirpath = tempfile.mkdtemp()
    cleanup = lambda: shutil.rmtree(dirpath)
    with cd(dirpath, cleanup):
        yield dirpath


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
        image_files = [f for f in archive]
    else:
        image_files = [fname]
    return image_files


def get_image(filepath):
    """Open image file as numpy array.

    Args:
        filepath: full filepath of image file

    Returns:
        img: numpy array of image data
    """
    logger.debug('Loading %s into numpy array', filepath)
    if os.path.splitext(filepath)[-1].lower() in {'.tif', '.tiff'}:
        img = skimage.external.tifffile.TiffFile(filepath).asarray()
        # tiff files should not have a channel dim
        img = np.expand_dims(img, axis=-1)
    else:
        img = keras_preprocessing.image.img_to_array(PIL.Image.open(filepath))

    logger.debug('Loaded %s into numpy array with shape %s',
                 filepath, img.shape)
    return img.astype('float32')


def pad_image(image, field):
    """Pad each the input image for proper dimensions when stitiching.

    Args:
        image: np.array of image data
        field: receptive field size of model

    Returns:
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
    """Split tensor into channels and save each as a tiff file.

    Args:
        arr: numpy array of image data
        name: name of original input image file
        subdir: optional subdirectory to save the result.
        output_dir: base directory for features

    Returns:
        out_paths: list of all saved image paths
    """
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

            skimage.external.tifffile.imsave(path, img)
            logger.debug('Saved channel %s to %s', channel, path)
            out_paths.append(path)
        except Exception as err:  # pylint: disable=broad-except
            out_paths = []
            logger.error('Could not save channel %s as image: %s',
                         channel, err)
    logger.debug('Saved %s image files in %s seconds.',
                 len(out_paths), timeit.default_timer() - start)
    return out_paths


# from deepcell.utils.tracking_utils.load_trks
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

    if filename.endswith(".zip"):
        with zipfile.ZipFile(filename, 'r') as zip_file:
            names = zip_file.namelist()
            images = {}

            for name in names:
                if name.endswith(".tiff") or name.endswith(".tif"):
                    img = np.float32(tiff.TiffFile(filepath).asarray())
                    img = np.expand_dims(img, axis=-1)

                if name.startswith("raw."):
                    images["X"] = img
                elif name.startswith("annotated."):
                    images["y"] = img

            return images

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

    try:
        logger.debug('Saving %s files to %s', len(files), filepath)
        with zipfile.ZipFile(filepath, 'w', allowZip64=True) as zip_file:
            for f in files:  # writing each file one by one
                name = f.replace(dest, '')
                name = name[1:] if name.startswith(os.path.sep) else name
                zip_file.write(f, arcname=name)
        logger.debug('Saved %s files to %s', len(files), filepath)
    except Exception as err:
        logger.error('Failed to write zipfile: %s', err)
        raise err
    logger.debug('Zipped %s files into %s in %s seconds.',
                 len(files), filepath, timeit.default_timer() - start)
    return filepath
