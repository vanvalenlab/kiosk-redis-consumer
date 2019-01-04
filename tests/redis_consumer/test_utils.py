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
"""Tests for utility functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile

from keras_preprocessing.image import array_to_img
from skimage.external import tifffile as tiff
import numpy as np
import pytest

from redis_consumer.pbs.predict_pb2 import PredictResponse
from redis_consumer.pbs import types_pb2
from redis_consumer.pbs.tensor_pb2 import TensorProto
from redis_consumer import utils
from redis_consumer import settings


def _get_image(img_h=300, img_w=300, channels=1):
    bias = np.random.rand(img_w, img_h, channels) * 64
    variance = np.random.rand(img_w, img_h, channels) * (255 - 64)
    img = np.random.rand(img_w, img_h, channels) * variance + bias
    return img


def _write_image(filepath, img_w=300, img_h=300):
    imarray = _get_image(img_h, img_w)
    if filepath.lower().endswith('tif') or filepath.lower().endswith('tiff'):
        tiff.imsave(filepath, imarray[..., 0])
    else:
        img = array_to_img(imarray, scale=False, data_format='channels_last')
        img.save(filepath)


def test_make_tensor_proto():
    # test with numpy array
    data = _get_image(300, 300, 1)
    proto = utils.make_tensor_proto(data, 'DT_FLOAT')
    assert isinstance(proto, (TensorProto,))
    # test with value
    data = 10.0
    proto = utils.make_tensor_proto(data, types_pb2.DT_FLOAT)
    assert isinstance(proto, (TensorProto,))


def test_grpc_response_to_dict():
    # TODO: how to fill up a dummy PredictResponse?
    response = PredictResponse()
    response_dict = utils.grpc_response_to_dict(response)
    assert isinstance(response_dict, (dict,))


def test_iter_image_archive():
    with utils.get_tempdir() as tempdir:
        zip_path = os.path.join(tempdir, 'test.zip')
        archive = zipfile.ZipFile(zip_path, 'w')
        num_files = 3
        for n in range(num_files):
            path = os.path.join(tempdir, '{}.tif'.format(n))
            _write_image(path, 30, 30)
            archive.write(path)
        archive.close()

        unzipped = [z for z in utils.iter_image_archive(zip_path, tempdir)]
        assert len(unzipped) == num_files


def test_get_image_files_from_dir():
    with utils.get_tempdir() as tempdir:
        zip_path = os.path.join(tempdir, 'test.zip')
        archive = zipfile.ZipFile(zip_path, 'w')
        num_files = 3
        for n in range(num_files):
            path = os.path.join(tempdir, '{}.tif'.format(n))
            _write_image(path, 30, 30)
            archive.write(path)
        archive.close()

        imfiles = utils.get_image_files_from_dir(path, None)
        assert len(imfiles) == 1

        imfiles = utils.get_image_files_from_dir(zip_path, tempdir)
        assert len(imfiles) == num_files


def test_get_image():
    with utils.get_tempdir() as temp_dir:
        # test tiff files
        test_img_path = os.path.join(temp_dir, 'phase.tif')
        _write_image(test_img_path, 300, 300)
        test_img = utils.get_image(test_img_path)
        np.testing.assert_equal(test_img.shape, (300, 300, 1))
        # test png files
        test_img_path = os.path.join(temp_dir, 'feature_0.png')
        _write_image(test_img_path, 400, 400)
        test_img = utils.get_image(test_img_path)
        # assert test_img.shape == 0
        np.testing.assert_equal(test_img.shape, (400, 400, 1))


def test_pad_image():
    # 2D images
    h, w = 300, 300
    img = _get_image(h, w)
    field_size = 61
    padded = utils.pad_image(img, field_size)

    new_h, new_w = h + (field_size - 1), w + (field_size - 1)
    np.testing.assert_equal(padded.shape, (new_h, new_w, 1))

    # 3D images
    frames = np.random.randint(low=1, high=6)
    imgs = np.vstack([_get_image(h, w)[None, ...] for i in range(frames)])
    padded = utils.pad_image(imgs, field_size)
    np.testing.assert_equal(padded.shape, (frames, new_h, new_w, 1))


def test_save_numpy_array():
    h, w = 30, 30
    c = np.random.randint(low=1, high=4)
    z = np.random.randint(low=1, high=6)

    with utils.get_tempdir() as tempdir:
        # 2D images
        img = _get_image(h, w, c)
        files = utils.save_numpy_array(img, 'name', '/a/b/', tempdir)
        assert len(files) == c
        for f in files:
            assert os.path.isfile(f)
            assert f.startswith(os.path.join(tempdir, 'a', 'b'))

        # 3D images
        imgs = np.vstack([_get_image(h, w, c)[None, ...] for i in range(z)])
        files = utils.save_numpy_array(imgs, 'name', '/a/b/', tempdir)
        assert len(files) == c
        for f in files:
            assert os.path.isfile(f)
            assert f.startswith(os.path.join(tempdir, 'a', 'b'))

    # Bad path will not fail, but will log error
    img = _get_image(h, w, c)
    files = utils.save_numpy_array(img, 'name', '/a/b/', '/does/not/exist/')
    assert len(files) == 0


def test_zip_files():
    n = np.random.randint(low=3, high=10)
    with utils.get_tempdir() as temp_dir:
        paths = [os.path.join(temp_dir, '{}.tif'.format(i)) for i in range(n)]
        for path in paths:
            _write_image(path, 30, 30)

        prefix = 'test'
        zip_path = utils.zip_files(paths, temp_dir, prefix)
        assert zip_path.startswith(temp_dir)
        assert os.path.basename(zip_path).startswith(prefix + '_')
        assert zipfile.is_zipfile(zip_path)

        with pytest.raises(Exception):
            bad_dest = os.path.join(temp_dir, 'does', 'not', 'exist')
            zip_path = utils.zip_files(paths, bad_dest, prefix)
