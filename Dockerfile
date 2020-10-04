FROM python:3.8
WORKDIR /usr/src/app
COPY requirements.txt ./

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
    python-dev \
    python3-dev \
    python-numpy \
    python3-pip\
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN pip install --no-cache-dir -r requirements.txt

COPY ./run.py .
COPY ./app ./app
COPY ./data ./data
COPY ./static ./static
COPY ./samples ./samples
COPY ./fonts ./fonts
RUN mkdir ./tmp

ENV environment=production

EXPOSE 5000

#CMD [ "python", "./run.py" ]
CMD ["flask", "run", "--host", "0.0.0.0"]