#! /bin/bash

DATA_DIR=${DATA_DIR:=".."}
WK_DIR=${WK_DIR:="."}

DS=Fluo-N3DH-CE
NUM=01
OUT=$DATA_DIR/$DS/${NUM}_RES
CFG="-cfg $DS/config.toml"

python normalize.py -lq 0.0001 -uq 0.9999 -z 10 \
    -o $WK_DIR/$DS/$NUM/image.zarr $DATA_DIR/$DS/$NUM/*.tif -ow

python inference.py \
    -os 64,0,64 -ts 240,512,360 -s 1,1,1 \
    -wp weights/$DS.ckpt \
    -o $WK_DIR/$DS/$NUM/image_pred_both.zarr $WK_DIR/$DS/$NUM/image.zarr -ow

rm data.db  # deleting previous database
ultrack segment $WK_DIR/$DS/$NUM/image_pred_both.zarr $CFG -r napari-dexp -el Boundary -dl Prediction
ultrack link $CFG
ultrack solve $CFG

rm -r $OUT
ultrack export ctc $CFG -o $OUT -s 0.1,1,1
