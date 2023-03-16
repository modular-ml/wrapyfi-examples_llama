FROM nvcr.io/nvidia/pytorch:23.02-py3

RUN apt-get update
RUN apt-get install -y git apt-utils
RUN  apt-get -yqq install ssh

RUN mkdir /repos/
WORKDIR /repos

RUN git clone https://github.com/fabawi/wrapyfi.git
RUN git clone https://github.com/modular-ml/wrapyfi-examples_llama.git

ENV LANG C.UTF-8
RUN cd wrapyfi/ && pip install .[pyzmq]

RUN cd wrapyfi-examples_llama/ && pip install -r requirements.txt && pip install -e .
