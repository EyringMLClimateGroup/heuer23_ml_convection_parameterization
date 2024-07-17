#!/bin/bash

# This script splits all 0001 files because these contain 2 timesteps in contrast to all other files

day=$1

files=$(find $1 -maxdepth 1 -name "*0001*nc")

for f in $files; do
    ./SplitFirstDataset.sh $f || exit 1
done
