FROM python:3.8
WORKDIR /usr/src/app

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-pip\
    zip
RUN apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN apt-get -qq -y install wkhtmltopdf

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt --upgrade

COPY ./run.py .
COPY ./app ./app
COPY ./data ./data
COPY ./static ./static
COPY ./fonts ./fonts
RUN mkdir /tmp/photo-id
