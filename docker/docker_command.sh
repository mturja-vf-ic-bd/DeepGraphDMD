#!/bin/bash

docker build -t dgdmd:v1.0 \
  --build-arg USER_ID=`id -u` \
  --build-arg GROUP_ID=`id -g` .
