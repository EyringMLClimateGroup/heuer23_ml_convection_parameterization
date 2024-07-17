#!/bin/bash

# This script splits one 0001 files because these contain 2 timesteps in contrast to all other files

if (( $# != 1 )); then
    echo "Usage: $0 0001_nc_file"
    exit 1
fi

if ! [[ $1 =~ 0001.*nc$ ]]; then
    echo "Please provide a nc file (timestep 0001)"
    exit 1
fi

echo "Splitting file $1"
filebase=${1/.nc/_}
cdo splitsel,1 $1 $filebase
echo "mv ${filebase}000001.nc ${1/0001/0000}"
mv ${filebase}000001.nc ${1/0001/0000}
echo "mv ${filebase}000002.nc $1"
mv ${filebase}000002.nc $1
echo "---done spliting---"
