# Copyright 2016-2022 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/kiosk-consumer/LICENSE
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
FROM python:3.8-slim-bullseye

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
    build-essential libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y git

COPY requirements.txt requirements-no-deps.txt ./

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps -r requirements-no-deps.txt

COPY . .

CMD ["/bin/sh", "-c", "python consume-redis-events.py"]
