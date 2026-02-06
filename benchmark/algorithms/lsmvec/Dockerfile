# benchmark/algorithms/lsmvec/Dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential cmake ninja-build git \
    && rm -rf /var/lib/apt/lists/*

# Python build deps
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install numpy pybind11 scikit-build-core

WORKDIR /workspace/big-ann-benchmarks
COPY . /workspace/big-ann-benchmarks

WORKDIR /workspace/big-ann-benchmarks/third_party/LSM-Vec
RUN python3 -m pip install .

WORKDIR /workspace/big-ann-benchmarks

