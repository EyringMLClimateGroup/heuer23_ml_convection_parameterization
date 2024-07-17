#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1            # Specify max. number of tasks to be invoked
#SBATCH --cpus-per-task=24      # Specify number of CPUs per task
#SBATCH --time=4:00:00        # Set a limit on the total run time
#SBATCH --mem=8000

if (( $# != 2 )); then
    echo "Need scratch_dir and work_dir"
    exit 1
fi

scratch_dir=$1
work_dir=$2

to_delete="${scratch_dir}"
to_delete_high_res="${work_dir}/ParamPrep/HighRes"
to_delete_virt_temp="${work_dir}/VirtPotTempRemapcon"
to_delete_tot_prec="${work_dir}/TotPrec"

rm -rf $to_delete_high_res
rm -rf $to_delete_virt_temp
rm -rf $to_delete_tot_prec
#tar czvf ${to_delete}.tgz $to_delete
#tar cf - $to_delete | pigz -9 -p 24 > ${to_delete}.tgz
rm -rf $to_delete
