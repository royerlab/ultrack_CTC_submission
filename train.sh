#!/usr/bin/sh

if [[ -z "$CTC_DIR" ]]; then
    echo "CTC variables not found"
    exit -1
fi

python train.py --n-epochs 20 --logdir $CTC_DIR/training/Fluo-C3DL-MDA231/logs/both $CTC_DIR/training/Fluo-C3DL-MDA231/tiles

python train.py --n-epochs 20 --logdir $CTC_DIR/training/Fluo-N3DH-CE/logs/both $CTC_DIR/training/Fluo-N3DH-CE/tiles
