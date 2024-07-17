#!/bin/bash

#SBATCH --account=bd1179
#SBATCH --partition=shared    # Specify partition name
#SBATCH --ntasks=1            # Specify max. number of tasks to be invoked
#SBATCH --cpus-per-task=2      # Specify number of CPUs per task
##SBATCH --time=1:00:00        # Set a limit on the total run time
##SBATCH --mem=65000
#SBATCH --time=00:30:00        # Set a limit on the total run time
#SBATCH --mem=80000
##SBATCH --array=0-112
##SBATCH --output=SlurmOut%A/%a.out
#SBATCH --job-name=high_res_conv_out

echo "Date: $(date)"

mamba run -n py3.8 python /work/bd1179/b309215/heuer23_ml_convection_parameterization/preprocessing/CalcConvOut.py $1 $2 $SLURM_ARRAY_TASK_ID
#CheckMemUse/mem_usage_while_proc_run.sh $! > slurm-mem_use_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt

py_exit_code=$?
exit $py_exit_code
