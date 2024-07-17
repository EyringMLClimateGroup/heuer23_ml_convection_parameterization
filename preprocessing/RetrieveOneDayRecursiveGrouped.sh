#!/bin/bash

date
day=$1

module unload slk
module load slk

target_folder="/scratch/b/b309215/HErZ-NARVALII/DATA/${day}"
mkdir -p ${target_folder}
lfs setstripe -E 1G -c 1 -S 1M -E 4G -c 4 -S 1M -E -1 -c 8 -S 1M ${target_folder}

# Use first variant for all days except 2013122400 (use second line)
slk_query=$(slk_helpers gen_file_query "/arch/bm0834/k203095/HErZ-NARVALII/DATA/${day}/(dei4_NARVALII?_${day}_fg_DOM01_ML_.*.nc|dei4_NARVALII?_${day}_cloud_DOM01_ML_.*.nc|dei4_NARVALII?_DOM01_${day}.tar)")
#slk_query=$(slk_helpers gen_file_query "/arch/bm0834/k203095/HErZ-NARVALII/DATA/${day}/(dei4_Barbados_${day}_fg_DOM01_ML_.*.nc|dei4_Barbados_${day}_cloud_DOM01_ML_.*.nc|dei4_NARVALII?_DOM01_${day}.tar)")

slk_search_id=$(eval "slk search '"${slk_query}"' | tail -n 1 | cut -c12-20")
if [[ ! -z $slk_search_id ]]; then
    echo "Retrieving files for search_id: $slk_search_id :"
    slk list $slk_search_id | cat
else
    echo No slk search id found $slk_search_id
    exit 1
fi

echo "Grouped:"
slk_helpers group_files_by_tape --search-id $slk_search_id --print-tape-id --print-tape-status
slk_grouping_exit_code=$?

for id in `slk_helpers group_files_by_tape --search-id $slk_search_id --run-search-query | awk ' { print $2 } '`; do
    echo "submitting search ${id}"
    sbatch --wait --account=bd1179 ./retrieve_slurm_search_id.sh $id $target_folder &
    pids[${id}]=$!
    #sleep 20 &
done

failed=0
for pid in ${pids[@]}; do
    wait $pid || { echo "job $pid failed"; failed=1; }
done

if (( $failed == 0 && $slk_grouping_exit_code == 0 )); then
    echo Executing: "find $target_folder -type f -exec mv {} $target_folder \; && find $target_folder -type d -empty -delete"
    find $target_folder -type f -exec mv {} $target_folder \; && find $target_folder -type d -empty -delete
    exit 0
else
    echo "Error: Some job failed"
    exit 1
fi
