FROM python:3.5

COPY ./requirements.txt /

RUN pip install -r requirements.txt

COPY ./redis-polling.py /

CMD ["/bin/sh", "-c", "python redis-polling.py"]
