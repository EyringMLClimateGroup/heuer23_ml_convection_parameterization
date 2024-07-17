#!/bin/bash

#SBATCH --account=bd1179
#SBATCH --partition=shared    # Specify partition name
#SBATCH --ntasks=1            # Specify max. number of tasks to be invoked
#SBATCH --cpus-per-task=4     # Specify number of CPUs per task
#SBATCH --time=1:00:00        # Set a limit on the total run time
#SBATCH --mem=10000
##SBATCH --array=0-34

##SBATCH --job-name=theta_v_remap

# Bind your OpenMP threads
export OMP_NUM_THREADS=4
export KMP_AFFINITY=verbose,granularity=thread,compact,1
export KMP_STACKSIZE=64m

# This script is for coarse-graining the virtual potential temperature and remapping back to source resolution

if (( $# != 4 )); then
    echo "Expecting 3 args: <source_region> <intermed_grid> <in_file_list> <output_dir>"
    exit 1
fi


source_reg="$1"
intermed_grid="$2"

files=($(cat ${3}))
in_path="${files[$SLURM_ARRAY_TASK_ID]}"
echo "In path: $in_path"
f_name=$(basename $in_path)

mkdir -p $4
out_path="$4/${f_name/.nc/}_intermed-${intermed_grid}_theta_v.nc"
echo "Final Out path: $out_path"

# Choosing source grid according to region DOM01 (R2B10) or DOM02 (R2B11)
if [[ $source_reg == "DOM01" ]]; then
    source_grid_path="/scratch/b/b309215/HErZ-NARVALII/GRIDS/narval_nestTropAtl_R2488m.nc"
elif [[ $source_reg == "DOM02" ]]; then
    source_grid_path="/scratch/b/b309215/HErZ-NARVALII/GRIDS/narval_nestTropAtl_R1250m.nc"
else
    echo "source grid not found"
    exit 1
fi

# Choosing target grid according to wanted coarse resolution
if [[ $intermed_grid == "R02B04" ]]; then
    intermed_grid_path="/pool/data/ICON/grids/public/mpim/0005/icon_grid_0005_R02B04_G.nc"
elif [[ $intermed_grid == "R02B05" ]]; then
    intermed_grid_path="/pool/data/ICON/grids/public/mpim/0019/icon_grid_0019_R02B05_G.nc"
elif [[ $intermed_grid == "R02B08" ]]; then
    intermed_grid_path="/pool/data/ICON/grids/public/mpim/0025/icon_grid_0025_R02B08_G.nc"
elif [[ $intermed_grid == "R02B09" ]]; then
    intermed_grid_path="/pool/data/ICON/grids/public/mpim/0016/icon_mask_0016_R02B09_G.nc"
else
    echo "target grid not found"
    exit 1
fi

source /home/${USER:0:1}/${USER}/.bashrc
mamba activate py3.9

cdo chname,theta_v,theta_v_e -remapcon,${source_grid_path} -remapcon,${intermed_grid_path} -setgrid,${source_grid_path} -selvar,theta_v $in_path $out_path
cdo_exit_code=$?

mamba deactivate
exit $cdo_exit_code