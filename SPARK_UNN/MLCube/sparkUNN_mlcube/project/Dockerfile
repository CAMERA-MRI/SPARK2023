# To try next : 11.0.3-devel-centos7 
# FROM nvidia/cuda:11.5.2-devel-ubuntu20.04
# to try FROM nvidia/cuda:11.6.2-base-ubuntu20.04
FROM nvidia/cuda:12.0.0-devel-ubuntu20.04
# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
# to remove questions like geographical location when installing python
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt-get update && apt-get install -y git 

RUN apt install --no-install-recommends  -y build-essential software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt install --no-install-recommends -y python3.9 python3-pip python3-setuptools python3-distutils 
RUN apt clean && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /mlcube_project/requirements.txt 

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir -r /mlcube_project/requirements.txt

ENV LANG C.UTF-8

COPY . /mlcube_project

ENTRYPOINT ["python3", "/mlcube_project/mlcube.py"]

