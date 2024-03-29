# ![DeepCell Kiosk Banner](https://raw.githubusercontent.com/vanvalenlab/kiosk-console/master/docs/images/DeepCell_Kiosk_Banner.png)

[![Build Status](https://github.com/vanvalenlab/kiosk-redis-consumer/workflows/build/badge.svg)](https://github.com/vanvalenlab/kiosk-redis-consumer/actions)
[![Coverage Status](https://coveralls.io/repos/github/vanvalenlab/kiosk-redis-consumer/badge.svg?branch=master)](https://coveralls.io/github/vanvalenlab/kiosk-redis-consumer?branch=master)
[![Documentation Status](https://readthedocs.org/projects/kiosk-redis-consumer/badge/?version=master)](https://deepcell-kiosk.readthedocs.io/projects/kiosk-redis-consumer/)
[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](/LICENSE)

The `kiosk-redis-consumer` reads events in Redis, downloads image data from the cloud, and sends the data to TensorFlow Serving via gRPC. The prediction is post-processed, zipped, and uploaded to the cloud.

This repository is part of the [DeepCell Kiosk](https://github.com/vanvalenlab/kiosk-console). More information about the Kiosk project is available through [Read the Docs](https://deepcell-kiosk.readthedocs.io/en/master) and our [FAQ](http://www.deepcell.org/faq) page.

## Custom Consumers

Custom consumers can be used to implement custom model pipelines. This documentation is a continuation of a [tutorial](https://deepcell-kiosk.readthedocs.io/en/master/CUSTOM-JOB.html) on building a custom job pipeline.

Consumers consume Redis events. Each type of Redis event is put into a queue (e.g. `predict`, `track`), and each queue has a specific consumer type will pop items off the queue.
Consumers call the `_consume` method to consume each item it finds in the queue.
This method must be implemented for every consumer.

The quickest way to get a custom consumer up and running is to:

1. Add a new file for the consumer: `redis_consumer/consumers/my_new_consumer.py`
2. Create a new class, inheriting from `TensorFlowServingConsumer` ([docs](https://deepcell-kiosk.readthedocs.io/projects/kiosk-redis-consumer/en/master/redis_consumer.consumers.html)), which uses the `preprocess`, `predict`, and `postprocess` methods to easily process data with the model.
3. Implement the `_consume` method, which should download the data, run inference on the data, save and upload the results, and finish the job by updating the Redis fields.
4. Import the new consumer in <tt><a href="https://github.com/vanvalenlab/kiosk-redis-consumer/blob/master/redis_consumer/consumers/__init__.py">redis_consumer/consumers/\_\_init\_\_.py</a></tt> and add it to the `CONSUMERS` dictionary with a correponding queue type (`queue_name`). The script <tt><a href="https://github.com/vanvalenlab/kiosk-redis-consumer/blob/master/consume-redis-events.py">consume-redis-events.py</a></tt> will load the consumer class based on the `CONSUMER_TYPE`.

See below for a basic implementation of `_consume()` making use of the methods inherited from `ImageFileConsumer`:

```python
def _consume(self, redis_hash):
    # get all redis data for the given hash
    hvals = self.redis.hgetall(redis_hash)

    # only work on unfinished jobs
    if hvals.get('status') in self.finished_statuses:
        self.logger.warning('Found completed hash `%s` with status %s.',
                            redis_hash, hvals.get('status'))
        return hvals.get('status')

    # Load input image
    fname = hvals.get('input_file_name')
    image = self.download_image(fname)

    # the model can be passed in as an environment variable,
    # and parsed in settings.py.
    model = 'NuclearSegmentation:1'

    # Use a custom Application from deepcell.applications
    app = self.get_grpc_app(model, deepcell.applications.NuclearSegmentation)

    # Run the predictions on the image
    results = app.predict(image)

    # save the results as an image file and upload it to the bucket
    save_name = hvals.get('original_name', fname)
    dest, output_url = self.save_output(image, redis_hash, save_name)

    # save the results to the redis hash
    self.update_key(redis_hash, {
        'status': self.final_status,
        'output_url': output_url,
        'upload_time': timeit.default_timer() - _,
        'output_file_name': dest,
        'total_jobs': 1,
        'total_time': timeit.default_timer() - start,
        'finished_at': self.get_current_timestamp()
    })

    # return the final status
    return self.final_status
```

For guidance on how to complete the deployment of a custom consumer, please return to [Tutorial: Custom Job](https://deepcell-kiosk.readthedocs.io/en/master/CUSTOM-JOB.html).

## Configuration

The consumer is configured using environment variables. Please find a table of all environment variables and their descriptions below.

| Name | Description | Default Value |
| :--- | :--- | :--- |
| `QUEUE` | **REQUIRED**: The Redis job queue to check for items to consume. | `"predict"` |
| `CONSUMER_TYPE` | **REQUIRED**: The type of consumer to run, used in `consume-redis-events.py`. | `"image"` |
| `STORAGE_BUCKET` | **REQUIRED**: The name of the storage bucket used to download and upload files. | `"s3://default-bucket"` |
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

Copyright © 2018-2022 [The Van Valen Lab](http://www.vanvalen.caltech.edu/) at the California Institute of Technology (Caltech), with support from the Paul Allen Family Foundation, Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
All rights reserved.
