#
# Darknet GPU Configuration File
# @author Frank Chude Qian (frankq at ieee dot com)
# v1.0.0
# 
# Copyright (c) 2020 Chude Qian - https://github.com/frank-qcd-qk
#

#! The base image from nvidia with cuda and cudnn
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

LABEL maintainer="frank1@ieee.org"

#! Set working directory
WORKDIR /workENV

#! Software install
RUN \
    apt-get update && apt-get install -y \
    gcc \
    git \
    git-lfs \
    vim \
    build-essential \
    python3 \
    python3-pip \
    libsm6 \ 
    libxext6 \
    libxrender-dev \ 
    wget && \
    rm -rf /var/lib/apt/lists/*


#! Copy core library
WORKDIR /workENV
COPY src/ ./src/

#! PIP time!

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN rm requirements.txt

#！ Test Nvidia
CMD ["/bin/sh "]

#docker container run -it testimage:1.2 /bin/bash