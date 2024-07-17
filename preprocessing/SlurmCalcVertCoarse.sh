#!/bin/bash
#=============================================================================
#SBATCH --account=bd1179

#SBATCH --partition=shared     # Specify partition name
#SBATCH --cpus-per-task=1      # Specify number of CPUs per task
#SBATCH --time=02:00:00        # Set a limit on the total run time

#SBATCH --job-name=vcg
#=============================================================================
echo "Date: $(date)"

if (( $# != 2 )); then
    echo "Expecting 2 args: <dir_for_coarse_graining> <R02B04 or R02B05 or R02B06>"
    exit 1
fi

files=($(ls $1/*.nc))
file=${files[$SLURM_ARRAY_TASK_ID]}
#file=${files[0]}

target_file_dir=$1/VertCoarse
mkdir -p $target_file_dir
in_file_name=$(basename $file)
target_file=${target_file_dir}/${in_file_name/.nc/_vertc.nc}

if (( $SLURM_NTASKS > 1 )); then
    PYTHON_SCRIPT="/work/bd1179/b309215/heuer23_ml_convection_parameterization/preprocessing/VerticalCoarseGrainingParallel.py"
elif (( $SLURM_NTASKS == 1 )); then
    PYTHON_SCRIPT="/work/bd1179/b309215/heuer23_ml_convection_parameterization/preprocessing/VerticalCoarseGraining.py"
else
    echo "Cannot work with SLURM_NTASKS: $SLURM_NTASKS"
    exit 1
fi

start_vcg="mamba run --no-capture-output -n py3.8 python $PYTHON_SCRIPT $2 $file $target_file $SLURM_NTASKS"
echo "Starting py script with $SLURM_NTASKS workers"

$start_vcg

py_exit_code=$?
echo "Exit code of $start_vcg: ${py_exit_code}"
echo "Done"
