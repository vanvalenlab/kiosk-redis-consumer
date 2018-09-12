FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    curl \
    python-pip \
    vim

COPY ./requirements.txt /

RUN pip install -r requirements.txt

COPY ./redis-polling.py /

#CMD sleep 100000
#CMD bash
#CMD ["python redis-polling.py"]
CMD python redis-polling.py
