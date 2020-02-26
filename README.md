kiosk-redis-consumer
====================

[![Build Status](https://travis-ci.org/vanvalenlab/kiosk-redis-consumer.svg?branch=master)](https://travis-ci.org/vanvalenlab/kiosk-redis-consumer)
[![Coverage Status](https://coveralls.io/repos/github/vanvalenlab/kiosk-redis-consumer/badge.svg?branch=master)](https://coveralls.io/github/vanvalenlab/kiosk-redis-consumer?branch=master)
[![Documentation Status](https://readthedocs.org/projects/kiosk-redis-consumer/badge/?version=master)](https://deepcell-kiosk.readthedocs.io/projects/kiosk-redis-consumer/en/latest/?badge=master)

Reads events in redis, downloads image data from the cloud, and send the data to TensorFlow Serving via gRPC.  The prediction is post-processed, zipped up, and uploaded to the cloud.

Custom Consumers
----------------

Custom consumers can be used to implement custom model pipelines. Check out our [tutorial](https://deepcell-kiosk.readthedocs.io/en/master/CUSTOM-JOB.html) on building a custom job pipeline.