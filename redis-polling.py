import os
import hashlib
import json
import logging
import time
import zipfile

import boto3
from decouple import config
from redis import StrictRedis
from skimage.external import tifffile as tiff
from tensorflow.python.keras.preprocessing.image import  img_to_array
from PIL import Image
import requests
import numpy as np


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
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')

for d in (DOWNLOAD_DIR, OUTPUT_DIR, LOG_DIR):
    try:
        os.mkdir(d)
    except OSError:
        pass

# configuring logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "log_file.txt"),
    format="%(asctime)s : %(name)s : %(levelname)s : %(message)s",
    level=logging.DEBUG)

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
redis = StrictRedis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    charset='utf-8')

# initialize S3 connection
s3 = boto3.client('s3',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)


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
    logging.debug("downloading")
    local_img = download_file(img_name)
    if local_img == "error":
        logging.debug("didn't download")
        output_file_location = "none"
        return output_file_location
    logging.debug("downloaded")

    logging.debug( "making array" )
    try:
        img = img_to_array(local_img)
    except Exception as err:
        logging.debug("didn't make array")
        errmsg = 'Could not read input image into numpy array: {}'.format(err)
        logging.debug(errmsg)
        output_file_location = "none"
        return output_file_location
        #raise ImageToArrayError(errmsg)
    logging.debug( "made array" )

    logging.debug( "sending to tf_serving" )
    tf_results = send_img_to_tfserving(img, model_name, version)
    if tf_results == "error":
        logging.debug("sent to, but didn't receive from tensorflow-serving")
        output_file_location = "none"
        return output_file_location
    logging.debug( "sent to tf_serving" )

    logging.debug( "saving" )
    out_paths = save_tf_serving_results(tf_results)
    if out_paths == "error":
        output_file_location = "none"
        return output_file_location

    zip_file = save_zip_file(out_paths)
    if zip_file == "error":
        output_file_location = "none"
        return output_file_location
    logging.debug( "saved" )

    logging.debug( "uploading" )
    try:
        upload_return_value = s3.upload_file(
            zip_file,
            AWS_S3_BUCKET,
            os.path.basename(zip_file))
    except Exception as err:
        logging.debug("didn't upload")
        errmsg = 'Failed to upload zipfile to S3 bucket: {}'.format(err)
        logging.error(errmsg)
        output_file_location = "none"
        return output_file_location
        #raise UploadFileError(errmsg)
    logging.debug( "uploaded" )

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
        logging.error(errmsg)
        local_image = "error"
        return local_image
        #raise S3DownloadError(errmsg)


def send_img_to_tfserving(img, model_name, version, tf_timeout=600):
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

        logging.debug("  http request sent")
        # Post to API URL
        prediction = requests.post(api_url, json=payload, timeout=tf_timeout)
        logging.debug("  http response received")

        logging.debug("  fixing json")
        # Fix JSON format (Temporary fix)
        prediction_fixed = fix_json(prediction)
        if prediction_fixed == "error":
            final_prediction = "error"
            return final_prediction
        logging.debug("  fixed json")

        # Check for Server errors
        if not prediction.status_code == 200:
            prediction_error = prediction.json()['error']
            raise TensorFlowServingError('{}: {}'.format(
                prediction_error, prediction.status_code))

        # Convert prediction to numpy array
        final_prediction =  np.array(list(prediction_fixed['predictions'][0]))
        return final_prediction
    except Exception as err:
        errmsg = 'Error during model prediction: {}'.format(err)
        logging.error(errmsg)
        final_prediction = "error"
        return final_prediction
        #raise TensorFlowServingError(errmsg)


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
        logging.error(errmsg)
        fixed_json = "error"
        return fixed_json
        #raise FixJsonError(errmsg)


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
            logging.error(errmsg)
            out_paths = "error"
            #raise SaveResultsError(errmsg)
    return out_paths


def save_zip_file(out_paths):
    try:
        pre_hashed_name = 'prediction_{}'.format(time.time()).encode('utf-8')
        hash_filename = '{}.zip'.format(hashlib.md5(pre_hashed_name).hexdigest())

        zip_filename = os.path.join(OUTPUT_DIR, hash_filename)

        # Create ZipFile and Write tiff files to it
        with zipfile.ZipFile(zip_filename, 'w') as zip_file:
            # writing each file one by one
            for out_file in out_paths:
                zip_file.write(out_file, arcname=os.path.basename(out_file))
        return zip_filename
    except Exception as err:
        errmsg = 'Failed to write zipfile: {}'.format(err)
        logging.error(errmsg)
        zip_filename = "error"
        return zip_filename
        #raise ZipFileException(errmsg)


def main():

    while True:

        # get all keys, accounting for the possibility that there are none
        try:
            all_keys = redis.keys()
            logging.debug("all_keys: %s", all_keys)
        except:
            all_keys = []

        # find the hashes from among the keys
        all_hashes = []
        for key in all_keys:
            key_type = redis.type(key)

            if key_type == "hash":
                all_hashes.append(key)

        logging.debug("all_hashes: %s", all_hashes)

        # look at each hash and decide whether or not to process it
        all_values = []
        for one_hash in all_hashes:
            #logging.debug("all_values: ")
            #logging.debug(all_values)
            #logging.debug("")
            hash_values = redis.hgetall(one_hash)
            img_name = one_hash
            url = hash_values['url']
            model_name = hash_values['model_name']
            model_version = hash_values['model_version']
            processing_status = hash_values['processed']
            logging.debug("current_image: %s", img_name)
            logging.debug("image_url: %s", url)
            logging.debug("model_name: %s", model_name)
            logging.debug("model_version: %s", model_version)
            logging.debug("processing_status: %s", processing_status)
            if processing_status == "no":
                # this image has not yet been claimed by any Tensorflow-serving instance
                # let's process it
                hset_response = redis.hset( one_hash, 'processed', 'processing' )
                logging.debug("processing image:")
                logging.debug( img_name )
                new_image_path = process_image( img_name, url, model_name, model_version )
                logging.debug("new_image_path: %s", new_image_path)
                logging.debug(" ")
                hset_response = redis.hset( one_hash, 'output_url', new_image_path )
                hset_response = redis.hset( one_hash, 'processed', 'yes' )

            all_values.append(hash_values)

        logging.debug("")
        logging.debug("all_values: %s", all_values)
        logging.debug("")

    time.sleep(10)


if __name__=='__main__':
    main()
