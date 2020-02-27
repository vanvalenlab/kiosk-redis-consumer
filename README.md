# kiosk-redis-consumer

[![Build Status](https://travis-ci.org/vanvalenlab/kiosk-redis-consumer.svg?branch=master)](https://travis-ci.org/vanvalenlab/kiosk-redis-consumer)
[![Coverage Status](https://coveralls.io/repos/github/vanvalenlab/kiosk-redis-consumer/badge.svg?branch=master)](https://coveralls.io/github/vanvalenlab/kiosk-redis-consumer?branch=master)
[![Documentation Status](https://readthedocs.org/projects/kiosk-redis-consumer/badge/?version=master)](https://deepcell-kiosk.readthedocs.io/projects/kiosk-redis-consumer/en/latest/?badge=master)

Reads events in redis, downloads image data from the cloud, and send the data to TensorFlow Serving via gRPC.  The prediction is post-processed, zipped up, and uploaded to the cloud.

## Custom Consumers

Custom consumers can be used to implement custom model pipelines. This documentation is a continuation of a [tutorial](https://deepcell-kiosk.readthedocs.io/en/master/CUSTOM-JOB.html) on building a custom job pipeline.

Consumers consume redis events. Each type of redis event is put into a separate queue (e.g. `predict`, `track`), and each consumer type will pop items to consume off that queue.

Each redis event should have the following fields:

- `model_name` - The name of the model that will be retrieved by TensorFlow Serving from `gs://<bucket-name>/models`
- `model_version` - The version number of the model in TensorFlow Serving
- `input_file_name` - The path to the data file in a cloud bucket.

If the consumer will send data to a TensorFlow Serving model, it should inherit from `redis_consumer.consumers.TensorFlowServingConsumer` ([docs](https://deepcell-kiosk.readthedocs.io/projects/kiosk-redis-consumer/en/master/redis_consumer.consumers.html)), which has methods `_get_predict_client()` and `grpc_image()` which can send data to the specific model.  The new consumer must also implement the `_consume()` method which performs the bulk of the work. The `_consume()` method will fetch data from redis, download data file from the bucket, process the data with a model, and upload the results to the bucket again. See below for a basic implementation of `_consume()`:

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

Finally, the new consumer needs to be registered in the script <tt><a href="https://github.com/vanvalenlab/kiosk-redis-consumer/blob/master/consume-redis-events.py">consume-redis-events.py</a></tt> by modifying the function `get_consumer()` shown below. Add a new if statement for the new queue type (`queue_name`) and the corresponding consumer.

```python
    def get_consumer(consumer_type, **kwargs):
        logging.debug('Getting `%s` consumer with args %s.', consumer_type, kwargs)
        ct = str(consumer_type).lower()
        if ct == 'image':
            return redis_consumer.consumers.ImageFileConsumer(**kwargs)
        if ct == 'zip':
            return redis_consumer.consumers.ZipFileConsumer(**kwargs)
        if ct == 'tracking':
            return redis_consumer.consumers.TrackingConsumer(**kwargs)
        raise ValueError('Invalid `consumer_type`: "{}"'.format(consumer_type))
```

For guidance on how to complete the deployment of a custom consumer, please return to [Tutorial: Custom Job](https://deepcell-kiosk.readthedocs.io/en/master/CUSTOM-JOB.html).

## Configuration

The consumer is configured using environment variables. Please find a table of all environment variables and their descriptions below.

| Name | Description | Default Value |
| :--- | :--- | :--- |
| `QUEUE` | The Redis job queue to check for items to consume. | `"predict"` |
| `CONSUMER_TYPE` | **REQUIRED**: The type of consumer to run, used in `consume-redis-events.py`. | `"image"` |
| `CLOUD_PROVIDER` | **REQUIRED**: The cloud provider, one of `"aws"` and `"gke"`. | `"gke"` |
| `GCLOUD_STORAGE_BUCKET` | **REQUIRED**: The storage bucket used to download and upload files. | `"default-bucket"` |
| `INTERVAL` | How frequently the consumer checks the Redis queue for items, in seconds. | `5` |
| `REDIS_HOST` | The IP address or hostname of Redis. | `"redis-master"` |
| `REDIS_PORT` | The port used to connect to Redis. | `6379` |
| `REDIS_TIMEOUT` | Timeout for each Redis request, in seconds. | `3` |
| `EMPTY_QUEUE_TIMEOUT` | Time to wait after finding an empty queue, in seconds. | `5` |
| `EXPIRE_TIME` | Expire Redis items this many seconds after completion. | `3600` |
| `TF_HOST` | The IP address or hostname of TensorFlow Serving. | `"tf-serving"` |
| `TF_PORT` | The port used to connect to TensorFlow Serving. | `8500` |
| `TF_TENSOR_NAME` | Name of input tensor for the exported model. | `"image"` |
| `TF_TENSOR_DTYPE` | The `dtype` used for the exported model. | `"DT_FLOAT"` |
| `GRPC_TIMEOUT` | Timeout for gRPC API requests, in seconds. | `30` |
| `GRPC_BACKOFF` | Time to wait before retrying a gRPC API request. | `3` |
| `MAX_RETRY` | Maximum number of retries for a failed TensorFlow Serving request. | `5` |
