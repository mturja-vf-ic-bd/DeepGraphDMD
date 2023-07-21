FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04
ARG USER_ID
ARG GROUP_ID

RUN apt-get update && apt-get install -y python3 python3-distutils python3-pip python3-apt

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

ENV HOME=/home/user
RUN chmod 777 $HOME

RUN mkdir -p $HOME/DeepGraphDMD
COPY requirements.txt $HOME/DeepGraphDMD/
WORKDIR $HOME/DeepGraphDMD
RUN pip3 install -r requirements.txt
