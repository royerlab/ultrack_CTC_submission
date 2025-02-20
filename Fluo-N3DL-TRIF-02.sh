#! /bin/bash

DATA_DIR=${DATA_DIR:=".."}
WK_DIR=${WK_DIR:="."}

DS=Fluo-N3DL-TRIF
NUM=02
OUT=$DATA_DIR/$DS/${NUM}_RES
CSB_OUT=$DATA_DIR/$DS/CSB/${NUM}_RES
CFG="-cfg $DS/config.toml"

python normalize.py -lq 0.5 -uq 0.99999 -z 1 \
    -o $WK_DIR/$DS/$NUM/image.zarr $DATA_DIR/$DS/$NUM/*.tif -ow

python compute_labels.py \
    -s1 0,1,1 -s2 1,4,4 -m 50 -M 1000 -c 500 -pt 0.5 \
    $WK_DIR/$DS/$NUM/image.zarr -ow

rm data.db  # deleting previous database
ultrack segment $WK_DIR/$DS/$NUM/image.zarr $CFG -r napari-dexp -el Boundary -dl Prediction
python register.py $WK_DIR/$DS/$NUM/image.zarr $CFG -r napari-dexp
ultrack link $CFG
ultrack solve $CFG

rm -r $OUT
ultrack export ctc $CFG -o $OUT \
    --first-frame-path $DATA_DIR/$DS/${NUM}_GT/TRA/man_track000.tif

rm -r $CSB_OUT
mkdir -p $CSB_OUT
ultrack export ctc $CFG -o $CSB_OUT
