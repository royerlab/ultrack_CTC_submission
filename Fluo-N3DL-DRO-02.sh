#! /bin/bash

DATA_DIR=${DATA_DIR:=".."}
WK_DIR=${WK_DIR:="."}

DS=Fluo-N3DL-DRO
NUM=02
OUT=$DATA_DIR/$DS/${NUM}_RES
CSB_OUT=$DATA_DIR/$DS/CSB/${NUM}_RES
CFG="-cfg $DS/config.toml"

python normalize.py -lq 0.0001 -uq 0.9999 -z 5 -out \
    -o $WK_DIR/$DS/$NUM/image.zarr $DATA_DIR/$DS/$NUM/*.tif -ow

python compute_labels.py \
    -s1 2,2,2 -s2 8,8,8 -m 100 -M 1000 -c 100 -pt 0.5 \
    $WK_DIR/$DS/$NUM/image.zarr -ow

python vector_field.py $WK_DIR/$DS/$NUM/image.zarr -o $WK_DIR/$DS/$NUM/vector.zarr -ow

rm data.db  # deleting previous database
ultrack segment $WK_DIR/$DS/$NUM/image.zarr $CFG -r napari-dexp -el Boundary -dl Prediction
ultrack add_flow $CFG $WK_DIR/$DS/$NUM/vector.zarr -r napari -cha 1
ultrack link $CFG
ultrack solve $CFG

rm -r $OUT
ultrack export ctc $CFG -o $OUT \
    -s 0.2,1,1 -di 1 --first-frame-path $DATA_DIR/$DS/${NUM}_GT/TRA/man_track000.tif

rm -r $CSB_OUT
mkdir -p $CSB_OUT
ultrack export ctc $CFG -o $CSB_OUT -s 0.2,1,1
