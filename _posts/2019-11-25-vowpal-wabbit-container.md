---
title: Running Vowpal Wabbit in Docker
layout: single
author_profile: false
mathjax: true
---

### Install and run Vowpal Wabbit in a Docker container.

---

### Background

From the official site

> Vowpal Wabbit provides fast, efficient, and flexible online machine learning techniques for reinforcement learning, supervised learning, and more. It is influenced by an ecosystem of community contributions, academic research, and proven algorithms. Vowpal Wabbit is sponsored by [Microsoft Research ](https://www.microsoft.com/en-us/research/lab/microsoft-research-new-york/).

I spent more then half a day trying to install Vowpal Wabbit on my native Anaconda installation before giving up and implementing the solution below. 

### Prerequisites

* [Docker](https://docs.docker.com/install/)

### Dockerfile

The Dockerfile is simple, but getting the dependencies right was not.

```
FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
        cmake \
        libboost-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libboost-math-dev \
        libboost-test-dev \
        zlib1g-dev \
        python3 \
        python3-pip \
        libboost-python-dev \
        git && \
        apt-get clean && apt-get autoclean

RUN git clone --recursive https://github.com/VowpalWabbit/vowpal_wabbit.git

WORKDIR vowpal_wabbit

RUN make

RUN pip3 install vowpalwabbit
```

Then simply run `docker build -t vowpal -f Dockerfile .`  and when building is done `docker run -it vowpal:latest /bin/bash` will allow you to run Vowpal Wabbit:

```
from vowpalwabbit import pyvw
vw = pyvw.vw(quiet=True)
ex = vw.example('1 | a b c')
vw.learn(ex)
vw.predict(ex)
0.632030725479126
```

