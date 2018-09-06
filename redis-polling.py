from decouple import config
from redis import Redis
import time

#import pdb
#from skimage import io
import boto3

from tensorflow.python.keras.preprocessing.image import  img_to_array
import requests
import numpy as np
import os
from skimage.external import tifffile as tiff
import hashlib
import zipfile

from PIL import Image

import json

# initializing environmental variables
DEBUG = config('DEBUG', default=True, cast=bool)
TF_HOST = config('TF_HOST', default='tf-serving-service')
TF_PORT = config('TF_PORT', default=1337, cast=int)
REDIS_HOST = config('REDIS_HOST', default='redis-master')
REDIS_PORT = config('REDIS_PORT', default=6379, cast=int)
AWS_REGION = config('AWS_REGION', default='us-east-1')
AWS_S3_BUCKET = config('AWS_S3_BUCKET', default='default-bucket')
AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID', default='specify_me')
AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY', default="specify_me")

# Application Directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(ROOT_DIR, 'download')
try:
    os.mkdir(DOWNLOAD_DIR)
except OSError:
    pass
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
try:
    os.mkdir(OUTPUT_DIR)
except OSError:
    pass

# Custom exceptions for error handling
class S3DownloadError(Exception):
    pass
class ImageToArrayError(Exception):
    pass
class FixJsonError(Exception):
    pass
class TensorFlowServingError(Exception):
    pass
class SaveResultsError(Exception):
    pass
class ZipFileException(Exception):
    pass
class UploadFileError(Exception):
    pass

# initialize Redis connection
redis = Redis( host=REDIS_HOST, port=REDIS_PORT )

# initialize S3 connection
s3 = boto3.client('s3', \
    region_name = AWS_REGION, \
    aws_access_key_id = AWS_ACCESS_KEY_ID, \
    aws_secret_access_key = AWS_SECRET_ACCESS_KEY)


def process_image(img_name, img_url, model_name, version):
    """Wrapper function
    Downloads image from S3,
    Loads image to numpy array
    Sends array to tf serving
    Saves each tf serving prediction features as image
    Uploads zip file of all result images to S3
    Returns URL of zipfile in S3
    Each failure case has a custom error class to send helpful status codes
    """
    print("downloading")
    local_img = download_file(img_name)
    print("downloaded")

    print( "making array" )
    try:
        img = img_to_array(local_img)
    except Exception as err:
        errmsg = 'Could not read input image into numpy array: {}'.format(err)
        raise ImageToArrayError(errmsg)
    print( "made array" )

    print( "sending to tf_serving" )
    tf_results = send_img_to_tfserving(img, model_name, version)
    print( "sent to tf_serving" )

    print( "saving" )
    out_paths = save_tf_serving_results(tf_results)

    zip_file = save_zip_file(out_paths)
    print( "saved" )

    print( "uploading" )
    try:
        upload_return_value = s3.upload_file(zip_file, AWS_S3_BUCKET, zip_file)
    except Exception as err:
        errmsg = 'Failed to upload zipfile to S3 bucket: {}'.format(err)
        raise UploadFileError(errmsg)
    print( "uploaded" )

    output_file_location = 'https://s3.amazonaws.com/{}/{}'.format(AWS_S3_BUCKET, zip_file)
    return output_file_location


def download_file(image_name):
    """Download File from S3 Storage"""
    try:
        output_location = os.path.join( DOWNLOAD_DIR, image_name )
        download_return_value = s3.download_file( AWS_S3_BUCKET, image_name, output_location)
        local_image = Image.open(output_location)
        return local_image
    except Exception as err:
        errmsg = 'Could not download file from S3 bucket: {}'.format(err)
        raise S3DownloadError(errmsg)


def send_img_to_tfserving(img, model_name, version, tf_timeout=300):
    """
    Send img as numpy array to tensorflow-serving,
    return result as numpy array of features
    """
    try:
        api_url = get_tfserving_url(model_name, version)

        # Define payload to send to API URL
        payload = {
            'instances': [
                {
                    'image': img.tolist()
                }
            ]
        }

        # Post to API URL
        prediction = requests.post(api_url, json=payload, timeout=tf_timeout)

        # Fix JSON format (Temporary fix)
        prediction_fixed = fix_json(prediction)

        # Check for Server errors
        if not prediction.status_code == 200:
            prediction_error = prediction.json()['error']
            raise TensorFlowServingError('{}: {}'.format(
                prediction_error, prediction.status_code))

        # Convert prediction to numpy array
        return np.array(list(prediction_fixed['predictions'][0]))
    except Exception as err:
        errmsg = 'Error during model prediction: {}'.format(err)
        raise TensorFlowServingError(errmsg)


def get_tfserving_url(model_name, version):
    """Get API URL for TensorFlow Serving, based on model name and version"""
    return 'http://{}:{}/v1/models/{}/versions/{}:predict'.format(
        TF_HOST, TF_PORT, model_name, version)


def fix_json(response):
    """
    Sometimes TF Serving has strange scientific notation e.g. "1e5.0,"
    so convert the float-exponent to an integer for JSON parsing.
    """
    try:
        raw_text = response.text
        fixed_text = raw_text.replace('.0,', ',').replace('.0],', '],')
        fixed_json = json.loads(fixed_text)
        return fixed_json
    except Exception as err:
        errmsg = 'Failed to fix json response: {}'.format(str(err))
        raise FixJsonError(errmsg)


def save_tf_serving_results(tf_results):
    """
    Split complete prediction into components and save each as tiff file
    """
    out_paths = []
    for channel in range(tf_results.shape[-1]):
        try:
            img = tf_results[:, :, channel].astype('float32')

            path = os.path.join(OUTPUT_DIR, 'feature_{}.tif'.format(channel))
            tiff.imsave(path, img)
            out_paths.append(path)
        except Exception as err:
            errmsg = 'Could not save predictions as image: {}'.format(err)
            raise SaveResultsError(errmsg)
    return out_paths


def save_zip_file(out_paths):
    try:
        pre_hashed_name = 'prediction_{}'.format(time.time()).encode('utf-8')
        hash_filename = '{}.zip'.format(hashlib.md5(pre_hashed_name).hexdigest())

        # Create ZipFile and Write tiff files to it
        with zipfile.ZipFile(hash_filename, 'w') as zip_file:
            # writing each file one by one
            for out_file in out_paths:
                zip_file.write(out_file, arcname=os.path.basename(out_file))
        return hash_filename
    except Exception as err:
        errmsg = 'Failed to write zipfile: {}'.format(err)
        raise ZipFileException(errmsg)


def main():

    while True:

        # get all keys, accounting for the possibility that there are none
	try:
	    all_keys = redis.keys()
            print("all_keys: ")
            print( str(all_keys) )
        except:
	    all_keys = []

        # find the hashes from among the keys
        all_hashes = []
        for key in all_keys:
            key_type = redis.type(key)
            if key_type=="hash":
                all_hashes.append(key)
        print("all_hashes: ")
        print( str(all_hashes) )

        # look at each hash and decide whether or not to process it
	all_values = []
        for one_hash in all_hashes:
            print("all_values: ")
            print(all_values)
            print("")
            hash_values = redis.hgetall(one_hash)
            img_name = one_hash
            url = hash_values['url']
            model_name = hash_values['model_name']
            model_version = hash_values['model_version']
            processing_status = hash_values['processed']
            if processing_status=="no":
                # this image has not yet been claimed by any Tensorflow-serving instance
                # let's process it
                hset_response = redis.hset( one_hash, 'processed', 'processing' )
                print("processing image:")
                print( img_name )
                new_image_path = process_image( img_name, url, model_name, model_version )
                print(new_image_path)
                print(" ")
                hset_response = redis.hset( one_hash, 'output_url', new_image_path )
                hset_response = redis.hset( one_hash, 'processed', 'yes' )

            all_values.append(hash_values)

        print("all_values: ")
	print(all_values)
        print("")

	time.sleep(10)


if __name__=='__main__':
    main()
