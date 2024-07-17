#!/bin/bash

counter=0
#days=(01)
#for i in ${days[@]}; do
for i in {07..07}; do
    exit_code=1
    d=201312${i}00
    #d=201608${i}00
    target_dir="/scratch/b/b309215/HErZ-NARVALII/DATA/${d}"
    echo $target_dir
    while (( $exit_code != 0 )); do
        echo "`date +%T` - Starting retrieve job for day ${d} ; exit_code: $exit_code ; counter: $counter"
        ./RetrieveOneDayRecursiveGrouped.sh $d
        exit_code=$?
        f=$(find $target_dir -name *slkretrieve)
        if [[ ! -z $f ]]; then
            echo "Removing Files"
            rm $f
        fi
        file_count=$(ls $target_dir | wc -l)
        # Looking for 73 files as: 31 days x 2 file_types + 1 tar_archive
        if (( $file_count == 73 )); then
            echo "File count equals 73"
            #for f in $(find $target_dir -type f); do
                #mv $f ${target_dir}/
                #rm ...
            #done
            exit_code=0
        fi
        (( counter = $counter + 1 ))
        if (( $counter > 10 )); then
            echo "Exiting, over 10 iterations"
            exit 1
        fi
        sleep 60
    done
    if (( $exit_code == 0 )); then
        ./NarvalSlurmPipeline.sh $d
    fi
done
