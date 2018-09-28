FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    curl \
    python-pip \
    python3 \
    python3-pip \
    vim

COPY ./requirements.txt /

RUN pip3 install -r requirements.txt

COPY ./redis-polling.py /

CMD sleep 100000
#CMD bash
#CMD ["/bin/sh", "-c", "python3 redis-polling.py"]
#CMD python redis-polling.py
