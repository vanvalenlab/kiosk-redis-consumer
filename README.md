# ![DeepCell Kiosk Banner](https://raw.githubusercontent.com/vanvalenlab/kiosk-console/master/docs/images/DeepCell_Kiosk_Banner.png)

[![Build Status](https://travis-ci.org/vanvalenlab/kiosk-redis-consumer.svg?branch=master)](https://travis-ci.org/vanvalenlab/kiosk-redis-consumer)
[![Coverage Status](https://coveralls.io/repos/github/vanvalenlab/kiosk-redis-consumer/badge.svg?branch=master)](https://coveralls.io/github/vanvalenlab/kiosk-redis-consumer?branch=master)
[![Documentation Status](https://readthedocs.org/projects/kiosk-redis-consumer/badge/?version=master)](https://deepcell-kiosk.readthedocs.io/projects/kiosk-redis-consumer/)
[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](/LICENSE)

The `kiosk-redis-consumer` reads events in Redis, downloads image data from the cloud, and sends the data to TensorFlow Serving via gRPC. The prediction is post-processed, zipped, and uploaded to the cloud.

This repository is part of the [DeepCell Kiosk](https://github.com/vanvalenlab/kiosk-console). More information about the Kiosk project is available through [Read the Docs](https://deepcell-kiosk.readthedocs.io/en/master) and our [FAQ](http://www.deepcell.org/faq) page.

## Custom Consumers

Custom consumers can be used to implement custom model pipelines. This documentation is a continuation of a [tutorial](https://deepcell-kiosk.readthedocs.io/en/master/CUSTOM-JOB.html) on building a custom job pipeline.

Consumers consume Redis events. Each type of Redis event is put into a separate queue (e.g. `predict`, `track`), and each consumer type will pop items to consume off that queue.

Each Redis event should have the following fields:

- `model_name` - The name of the model that will be retrieved by TensorFlow Serving from `gs://<bucket-name>/models`
- `model_version` - The version number of the model in TensorFlow Serving
- `input_file_name` - The path to the data file in a cloud bucket.

If the consumer will send data to a TensorFlow Serving model, it should inherit from `redis_consumer.consumers.TensorFlowServingConsumer` ([docs](https://deepcell-kiosk.readthedocs.io/projects/kiosk-redis-consumer/en/master/redis_consumer.consumers.html)), which has methods `_get_predict_client()` and `grpc_image()` which can send data to the specific model.  The new consumer must also implement the `_consume()` method which performs the bulk of the work. The `_consume()` method will fetch data from Redis, download data file from the bucket, process the data with a model, and upload the results to the bucket again. See below for a basic implementation of `_consume()`:

```python
    def _consume(self, redis_hash):
        # get all redis data for the given hash
        hvals = self.redis.hgetall(redis_hash)

        with utils.get_tempdir() as tempdir:
            # download the image file
            fname = self.storage.download(hvals.get('input_file_name'), tempdir)

            # load image file as data
            image = utils.get_image(fname)

            # preprocess data if necessary

            # send the data to the model
            results = self.grpc_image(image,
                                    hvals.get('model_name'),
                                    hvals.get('model_version'))

            # postprocess results if necessary

            # save the results as an image
            outpaths = utils.save_numpy_array(results, name=name,
                                            subdir=subdir, output_dir=tempdir)

            # zip up the file
            zip_file = utils.zip_files(outpaths, tempdir)

            # upload the zip file to the cloud bucket
            dest, output_url = self.storage.upload(zip_file)

            # save the results to the redis hash
            self.update_key(redis_hash, {
                'status': self.final_status,
                'output_url': output_url,
                'output_file_name': dest
                })

        # return the final status
        return self.final_status
```

Finally, the new consumer needs to be imported into the <tt><a href="https://github.com/vanvalenlab/kiosk-redis-consumer/blob/master/redis_consumer/consumers/__init__.py">redis_consumer/consumers/\_\_init\_\_.py</a></tt> and added to the `CONSUMERS` dictionary with a correponding queue type (`queue_name`). The script <tt><a href="https://github.com/vanvalenlab/kiosk-redis-consumer/blob/master/consume-redis-events.py">consume-redis-events.py</a></tt> will load the consumer class based on the `CONSUMER_TYPE`.

```python
# Custom Workflow consumers
from redis_consumer.consumers.image_consumer import ImageFileConsumer
from redis_consumer.consumers.tracking_consumer import TrackingConsumer
# TODO: Import future custom Consumer classes.


CONSUMERS = {
    'image': ImageFileConsumer,
    'zip': ZipFileConsumer,
    'tracking': TrackingConsumer,
    # TODO: Add future custom Consumer classes here.
}
```

For guidance on how to complete the deployment of a custom consumer, please return to [Tutorial: Custom Job](https://deepcell-kiosk.readthedocs.io/en/master/CUSTOM-JOB.html).

## Configuration

The consumer is configured using environment variables. Please find a table of all environment variables and their descriptions below.

| Name | Description | Default Value |
| :--- | :--- | :--- |
| `QUEUE` | **REQUIRED**: The Redis job queue to check for items to consume. | `"predict"` |
| `CONSUMER_TYPE` | **REQUIRED**: The type of consumer to run, used in `consume-redis-events.py`. | `"image"` |
| `CLOUD_PROVIDER` | **REQUIRED**: The cloud provider, one of `"aws"` and `"gke"`. | `"gke"` |
| `GCLOUD_STORAGE_BUCKET` | **REQUIRED**: The name of the storage bucket used to download and upload files. | `"default-bucket"` |
| `INTERVAL` | How frequently the consumer checks the Redis queue for items, in seconds. | `5` |
| `REDIS_HOST` | The IP address or hostname of Redis. | `"redis-master"` |
| `REDIS_PORT` | The port used to connect to Redis. | `6379` |
| `REDIS_TIMEOUT` | Timeout for each Redis request, in seconds. | `3` |
| `EMPTY_QUEUE_TIMEOUT` | Time to wait after finding an empty queue, in seconds. | `5` |
| `DO_NOTHING_TIMEOUT` | Time to wait after finding an item that requires no work, in seconds. | `0.5` |
| `STORAGE_MAX_BACKOFF` | Maximum time to wait before retrying a Storage request | `60` |
| `EXPIRE_TIME` | Expire Redis items this many seconds after completion. | `3600` |
| `METADATA_EXPIRE_TIME` | Expire cached model metadata after this many seconds. | `30` |
| `TF_HOST` | The IP address or hostname of TensorFlow Serving. | `"tf-serving"` |
| `TF_PORT` | The port used to connect to TensorFlow Serving. | `8500` |
| `GRPC_TIMEOUT` | Timeout for gRPC API requests, in seconds. | `30` |
| `GRPC_BACKOFF` | Time to wait before retrying a gRPC API request. | `3` |
| `MAX_RETRY` | Maximum number of retries for a failed TensorFlow Serving request. | `5` |

## Contribute

We welcome contributions to the [kiosk-console](https://github.com/vanvalenlab/kiosk-console) and its associated projects. If you are interested, please refer to our [Developer Documentation](https://deepcell-kiosk.readthedocs.io/en/master/DEVELOPER.html), [Code of Conduct](https://github.com/vanvalenlab/kiosk-console/blob/master/CODE_OF_CONDUCT.md) and [Contributing Guidelines](https://github.com/vanvalenlab/kiosk-console/blob/master/CONTRIBUTING.md).

## License

This software is license under a modified Apache-2.0 license. See [LICENSE](/LICENSE) for full  details.

## Copyright

Copyright © 2018-2020 [The Van Valen Lab](http://www.vanvalen.caltech.edu/) at the California Institute of Technology (Caltech), with support from the Paul Allen Family Foundation, Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
All rights reserved.
