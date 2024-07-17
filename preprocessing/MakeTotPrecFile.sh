#!/bin/bash

# This script extracts an narval tar archive and creates a file containing the total precipitation data for that archive

if (( $# != 2 )); then
    echo "Provide 2 arguments: $0 <source_dir> <target_dir>"
    echo "source_dir has to be the directory where a tar file in the format dei4_NARVALII_DOM01_[0-9]*.tar exists"
    echo "target_dir has to be the directory where tot_prec file is written in a subfolder TotPrec"
    exit 1
fi

file=$(find $1 -regex ".*/dei4_NARVALII_DOM01_[0-9]*.tar")

regex=".*(20[0-2][0-9][0-9][0-9][0-9][0-9]00).*"
[[ $file =~ $regex ]]
year="${BASH_REMATCH[1]}"

dir=${file/.tar/}
mkdir $dir
tar xvf $file -C $dir

tot_prec_dir="$2/TotPrec"
mkdir -p $tot_prec_dir
tot_prec_file="${tot_prec_dir}/dei4_NARVALII_${year}_DOM01_ML_tot_prec.nc"

cdo -mergetime -apply,-selvar,tot_prec [ ${dir}/*.nc ] ${tot_prec_file}
