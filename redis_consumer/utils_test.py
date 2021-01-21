# Copyright 2016-2020 The Van Valen Lab at the California Institute of
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
import tarfile
import tempfile
import zipfile

import pytest

import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img
from skimage.external import tifffile as tiff

from redis_consumer.testing_utils import _get_image

from redis_consumer import utils
from redis_consumer import settings


def _write_image(filepath, img_w=300, img_h=300):
    imarray = _get_image(img_h, img_w)
    if filepath.lower().endswith('tif') or filepath.lower().endswith('tiff'):
        tiff.imsave(filepath, imarray[..., 0])
    else:
        img = array_to_img(imarray, scale=False, data_format='channels_last')
        img.save(filepath)


def _write_trks(filepath, X_mean=10, y_mean=5,
                img_w=300, img_h=300, channels=1, frames=30):
    raw = X_mean + np.random.rand(frames, img_w, img_h, channels) - 0.5
    tracked = y_mean + np.random.rand(frames, img_w, img_h, channels) - 0.5
    with tarfile.open(filepath, 'w') as trks:
        with tempfile.NamedTemporaryFile() as raw_file:
            np.save(raw_file, raw)
            raw_file.flush()
            trks.add(raw_file.name, 'raw.npy')

        with tempfile.NamedTemporaryFile() as tracked_file:
            np.save(tracked_file, tracked)
            tracked_file.flush()
            trks.add(tracked_file.name, 'tracked.npy')


def test_iter_image_archive(tmpdir):
    tmpdir = str(tmpdir)
    zip_path = os.path.join(tmpdir, 'test.zip')
    archive = zipfile.ZipFile(zip_path, 'w')
    num_files = 3
    for n in range(num_files):
        path = os.path.join(tmpdir, '{}.tif'.format(n))
        _write_image(path, 30, 30)
        archive.write(path)
    archive.close()

    unzipped = [z for z in utils.iter_image_archive(zip_path, tmpdir)]
    assert len(unzipped) == num_files


def test_get_image_files_from_dir(tmpdir):
    tmpdir = str(tmpdir)

    zip_path = os.path.join(tmpdir, 'test.zip')
    archive = zipfile.ZipFile(zip_path, 'w')
    num_files = 3
    for n in range(num_files):
        path = os.path.join(tmpdir, '{}.tif'.format(n))
        _write_image(path, 30, 30)
        archive.write(path)
    archive.close()

    imfiles = list(utils.get_image_files_from_dir(path, None))
    assert len(imfiles) == 1

    imfiles = list(utils.get_image_files_from_dir(zip_path, tmpdir))
    assert len(imfiles) == num_files


def test_get_image(tmpdir):
    tmpdir = str(tmpdir)
    # test tiff files
    test_img_path = os.path.join(tmpdir, 'phase.tif')
    _write_image(test_img_path, 300, 300)
    test_img = utils.get_image(test_img_path)
    print(test_img.shape)
    np.testing.assert_equal(test_img.shape, (300, 300, 1))
    # test png files
    test_img_path = os.path.join(tmpdir, 'feature_0.png')
    _write_image(test_img_path, 400, 400)
    test_img = utils.get_image(test_img_path)
    # assert test_img.shape == 0
    np.testing.assert_equal(test_img.shape, (400, 400, 1))


def test_save_numpy_array(tmpdir):
    tmpdir = str(tmpdir)
    h, w = 30, 30
    c = np.random.randint(low=1, high=4)
    z = np.random.randint(low=1, high=6)

    # 2D images without channel axis
    img = _get_image(h, w, 1)
    img = np.squeeze(img)
    files = utils.save_numpy_array(img, 'name', '/a/b/', tmpdir)
    assert len(files) == 1
    for f in files:
        assert os.path.isfile(f)
        assert f.startswith(os.path.join(tmpdir, 'a', 'b'))

    # 2D images
    img = _get_image(h, w, c)
    files = utils.save_numpy_array(img, 'name', '/a/b/', tmpdir)
    assert len(files) == c
    for f in files:
        assert os.path.isfile(f)
        assert f.startswith(os.path.join(tmpdir, 'a', 'b'))

    # 3D images
    imgs = np.vstack([_get_image(h, w, c)[None, ...] for i in range(z)])
    files = utils.save_numpy_array(imgs, 'name', '/a/b/', tmpdir)
    assert len(files) == c
    for f in files:
        assert os.path.isfile(f)
        assert f.startswith(os.path.join(tmpdir, 'a', 'b'))

    # Bad path will not fail, but will log error
    img = _get_image(h, w, c)
    files = utils.save_numpy_array(img, 'name', '/a/b/', '/does/not/exist/')
    assert not files


def test_load_track_file(tmpdir):
    tmpdir = str(tmpdir)
    for i in range(10):
        # random boolean
        if np.random.choice([True, False]):
            trk_file = '{}.trk'.format(i)
        else:
            trk_file = '{}.trks'.format(i)

        trk_file = os.path.join(tmpdir, trk_file)

        w = np.random.randint(low=30, high=100)
        h = np.random.randint(low=30, high=100)
        f = np.random.randint(low=2, high=30)

        X_mean = np.random.uniform(0, 10)
        y_mean = np.random.uniform(0, 10)

        _write_trks(trk_file, X_mean=X_mean, y_mean=y_mean,
                    img_w=w, img_h=h, channels=1, frames=f)

        trks = utils.load_track_file(trk_file)

        assert "X" in trks
        assert "y" in trks
        assert len(trks) == 2

        assert ((X_mean - 0.5 < trks["X"]).all() and
                (trks["X"] < X_mean + 0.5).all())
        assert ((y_mean - 0.5 < trks["y"]).all() and
                (trks["y"] < y_mean + 0.5).all())

        assert trks["X"].shape == (f, w, h, 1)
        assert trks["y"].shape == (f, w, h, 1)

    # test bad extension
    with pytest.raises(Exception):
        path = os.path.join(tmpdir, "non.bad_extension")
        _write_trks(path)
        trks = utils.load_track_file(path)

    # test non-existent file
    with pytest.raises(Exception):
        path = os.path.join(tmpdir, "poof.trk")
        trks = utils.load_track_file(path)


def test_zip_files(tmpdir):
    n = np.random.randint(low=3, high=10)
    tmpdir = str(tmpdir)

    paths = [os.path.join(tmpdir, '{}.tif'.format(i)) for i in range(n)]
    for path in paths:
        _write_image(path, 30, 30)

    prefix = 'test'
    zip_path = utils.zip_files(paths, tmpdir, prefix)
    assert zip_path.startswith(tmpdir)
    assert os.path.basename(zip_path).startswith(prefix + '_')
    assert zipfile.is_zipfile(zip_path)

    with pytest.raises(Exception):
        bad_dest = os.path.join(tmpdir, 'does', 'not', 'exist')
        zip_path = utils.zip_files(paths, bad_dest, prefix)


def test__pick_model(mocker):
    mocker.patch.object(settings, 'MODEL_CHOICES', {0: 'dummymodel:0'})
    res = utils._pick_model(0)
    assert len(res) == 2
    assert res[0] == 'dummymodel'
    assert res[1] == '0'

    with pytest.raises(ValueError):
        utils._pick_model(-1)
