# kiosk-redis-consumer

[![Build Status](https://travis-ci.org/vanvalenlab/kiosk-redis-consumer.svg?branch=master)](https://travis-ci.org/vanvalenlab/kiosk-redis-consumer)
[![Coverage Status](https://coveralls.io/repos/github/vanvalenlab/kiosk-redis-consumer/badge.svg?branch=master)](https://coveralls.io/github/vanvalenlab/kiosk-redis-consumer?branch=master)

Reads events in redis, downloads image data from the cloud, and send the data to TensorFlow Serving via gRPC.  The prediction is post-processed, zipped up, and uploaded to the cloud.
