#!/bin/bash

#SBATCH --job-name=retr_slk_lost_files # Specify job name
#SBATCH --partition=compute # partition name
#SBATCH --ntasks=1 # max. number of tasks to be invoked
#SBATCH --time=05:00:00 # Set a limit on the total run time
#SBATCH --account=bd1179 # Charge resources on this project
#SBATCH --mem=8GB

# This script uses retrieves files which have not been retrieved in a `slk retrieve` call by first removing all .slkretrieve files and then comparing with the corresponding archive directory and consequent retrieval

date
# ~~~~~~~~~~~~ preparation ~~~~~~~~~~~~
if (( $# != 1 )); then
    echo "Need one input parameter. Usage: $0 <day>"
    exit 1
fi

module unload slk
module load slk

# set target folder for retrieval
day=$1
scratch_dir="/scratch/b/b309215/HErZ-NARVALII/DATA"
data_dir=${scratch_dir}/${day}
tmp_dir="/tmp/b309215/${day}/slk_lost_retr"

mkdir -p $tmp_dir

lost_file_paths=$(find $data_dir -name "*.slkretrieve")

for f in ${lost_file_paths[@]}; do
    rm $f
done

tmp_dir_wanted_files=${tmp_dir}/wanted_files.txt
tmp_dir_retr_files=${tmp_dir}/retr_files.txt
slk_query=$(slk_helpers gen_file_query "/arch/bm0834/k203095/HErZ-NARVALII/DATA/${day}/(dei4_NARVALII?_${day}_fg_DOM01_ML_.*.nc|dei4_NARVALII?_${day}_cloud_DOM01_ML_.*.nc|dei4_NARVALII?_DOM01_${day}.tar)")
slk_search_id=$(eval "slk_helpers search_limited '"${slk_query}"' | tail -n 1 | cut -c12-20")
slk list $slk_search_id | grep $day | awk '{printf $NF "\n"}' | xargs -n1 basename > ${tmp_dir_wanted_files}
ls ${data_dir} > ${tmp_dir_retr_files}
lost_files=$(sort $tmp_dir_wanted_files $tmp_dir_retr_files | uniq -u)

if [[ ! -z "${lost_files// }" ]]; then
    lost_files_regex=$(echo $lost_files | xargs -n 1 printf "%s|")
    lost_files_regex="(${lost_files_regex%?})"
    slk_query=$(slk_helpers gen_file_query /arch/bm0834/k203095/HErZ-NARVALII/DATA/${day}/${lost_files_regex})
    slk_search_id=$(eval "slk_helpers search_limited '"${slk_query}"' | tail -n 1 | cut -c12-20")
    echo "Retrieving files (search_id=$slk_search_id):"
    slk list $slk_search_id | cat

    #echo "Executing command: slk retrieve $slk_search_id $data_dir"
    #slk retrieve $slk_search_id $data_dir

    if [ $? -ne 0 ]; then
        >&2 echo "an error occurred in slk retrieve call of lost files for $day"
        exit 1
    else
        echo "retrieval of lost files successful for $day"
        exit 0
    fi
else
    echo "No lost files found"
    exit 0
fi
