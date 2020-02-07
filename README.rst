kiosk-redis-consumer
====================

.. image:: https://travis-ci.org/vanvalenlab/kiosk-redis-consumer.svg?branch=master
    :target: https://travis-ci.org/vanvalenlab/kiosk-redis-consumer
    :alt: Build Status

.. image:: https://coveralls.io/repos/github/vanvalenlab/kiosk-redis-consumer/badge.svg?branch=master
    :target: https://coveralls.io/github/vanvalenlab/kiosk-redis-consumer?branch=master
    :alt: Coverage Status

.. image:: https://readthedocs.org/projects/kiosk-redis-consumer/badge/?version=master
    :target: https://deepcell-kiosk.readthedocs.io/projects/kiosk-redis-consumer/en/latest/?badge=master
    :alt: Documentation Status

Reads events in redis, downloads image data from the cloud, and send the data to TensorFlow Serving via gRPC.  The prediction is post-processed, zipped up, and uploaded to the cloud.

Custom Consumers
----------------

Custom consumers can be used to implement custom model pipelines. Check out :doc:`kiosk:CUSTOM-JOB`.