#! /bin/bash

# copying config
files=(
    Fluo-C3DL-MDA231
    Fluo-N3DH-CE
    Fluo-N3DL-DRO
    Fluo-N3DL-TRIC
    Fluo-N3DL-TRIF
)

rm data.db
rm metadata.toml

for dir in "${files[@]}";
do
    rm -r $dir/01/*.zarr
    rm -r $dir/01/*.npy
    rm -r $dir/02/*.zarr
    rm -r $dir/02/*.npy
done;
