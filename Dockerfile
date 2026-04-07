FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive 

# install some essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
   lsb-release \
   build-essential \
   python3 python3-dev python3-pip \
   cmake \
   git \
   vim \
   ca-certificates \
   libzmqpp-dev \
   libopencv-dev \
   gnupg2 \
   && rm -rf /var/lib/apt/lists/*

# install python 3.9
RUN apt-get update && apt-get install -y software-properties-common curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.9 python3.9-dev python3.9-distutils \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 \
    && rm -rf /var/lib/apt/lists/*


RUN /bin/bash -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 

# install ROS Noetic
RUN apt-get update && apt-get install -y --no-install-recommends \
   ros-noetic-desktop-full 

# install catkin tools
RUN pip3 install setuptools catkin-tools 

COPY . /home/flightmare

ENV FLIGHTMARE_PATH=/home/flightmare

# install pytorch / torchvision with CUDA
RUN pip3 install torch torchvision

# install any other dependencies required
RUN pip3 install --ignore-installed psutil
RUN cd /home/flightmare/flightlib && pip3 install . \
    && cd /home/flightmare/flightrl && pip3 install .

# matplotlib errors 
RUN pip3 install matplotlib --upgrade