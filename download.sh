#! /bin/bash

VERSION=2023_10

# Download weights if they do not exist
wget -nc -P weights https://public.czbiohub.org/royerlab/ultrack/CTC/weights/$VERSION/Fluo-C3DL-MDA231.ckpt
wget -nc -P weights https://public.czbiohub.org/royerlab/ultrack/CTC/weights/$VERSION/Fluo-N3DH-CE.ckpt
