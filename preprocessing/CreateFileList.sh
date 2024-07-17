#!/bin/bash

ls /scratch/b/b309215/HErZ-NARVALII/DATA/${1}/dei4_NARVALI*_${1}_fg_DOM01_ML_*.nc > ./in_files_${1}.txt

if [ $? -ne 0 ]; then
    >&2 echo "error occured while creating file list"
    exit 1
else
    echo "created file list"
fi

# Looking for 37 files as we have 37 timesteps per day
if (( `cat /work/bd1179/b309215/heuer23_ml_convection_parameterization/preprocessing/in_files_${1}.txt* | wc -l` != 37 )); then
    echo "Not enough files found"
    exit 1
fi
