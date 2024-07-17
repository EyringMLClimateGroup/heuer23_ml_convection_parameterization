#!/bin/bash

#SBATCH --account=bd1179
#SBATCH --partition=shared    # Specify partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1            # Specify max. number of tasks to be invoked
#SBATCH --cpus-per-task=4      # Specify number of CPUs per task
#SBATCH --time=1:00:00        # Set a limit on the total run time
#SBATCH --mem=8000
##SBATCH --array=0-258
#SBATCH --job-name=hcg

# Bind your OpenMP threads
export OMP_NUM_THREADS=4
export KMP_AFFINITY=verbose,granularity=thread,compact,1
export KMP_STACKSIZE=64m

if (( $# != 3 )); then
    echo "Expecting 3 args: <source_region> <target_grid> <work_path>"
    exit 1
fi

source_reg=$1
target_grid=$2
work_path=$3

# Cropping region by 2° for either DOM01 (R2B10) or DOM02 (R2B11)
if [[ $source_reg == "DOM01" ]]; then
    source_grid_path="/scratch/b/b309215/HErZ-NARVALII/GRIDS/narval_nestTropAtl_R2488m.nc"
    box_m2degr="-66.6,13.9,-9.3,19.1"
elif [[ $source_reg == "DOM02" ]]; then
    source_grid_path="/scratch/b/b309215/HErZ-NARVALII/GRIDS/narval_nestTropAtl_R1250m.nc"
    box_m2degr="-63.2,-39.6,-3.1,17.1"
else
    echo "source grid not found"
    exit 1
fi

# Choosing target grid according to wanted coarse resolution
if [[ $target_grid == "R02B04" ]]; then
    target_grid_path="/pool/data/ICON/grids/public/mpim/0005/icon_grid_0005_R02B04_G.nc"
    weight_file="/work/bd1179/b309215/R02B04NarvalWeightFile/dei4_NARVALII_2016082500_fg_DOM01_ML_0001_conv_out_cropped_weights.nc"
elif [[ $target_grid == "R02B05" ]]; then
    target_grid_path="/pool/data/ICON/grids/public/mpim/0019/icon_grid_0019_R02B05_G.nc"
    weight_file="/work/bd1179/b309215/R02B05NarvalWeightFile/dei4_NARVALII_2016082500_fg_DOM01_ML_0001_conv_out_cropped_weights.nc"
elif [[ $target_grid == "R02B06" ]]; then
    target_grid_path="/pool/data/ICON/grids/public/mpim/0007/icon_grid_0007_R02B06_G.nc"
    weight_file="/work/bd1179/b309215/R02B06NarvalWeightFile/dei4_NARVALI_2013123100_fg_DOM01_ML_0000_conv_out_cropped_weights.nc"
elif [[ $target_grid == "R02B06_0021" ]]; then
    target_grid_path="/pool/data/ICON/grids/public/mpim/0021/icon_grid_0021_R02B06_G.nc"
    weight_file="/work/bd1179/b309215/R02B06_0021NarvalWeightFile/dei4_NARVALII_2016082500_fg_DOM01_ML_0001_conv_out_cropped_weights.nc"
elif [[ $target_grid == "R02B08" ]]; then
    target_grid_path="/pool/data/ICON/grids/public/mpim/0025/icon_grid_0025_R02B08_G.nc"
else
    echo "target grid not found"
    exit 1
fi

# Preparation of output

target_dir="${work_path}/ParamPrep/LowRes/$target_grid"
high_res_dir="${work_path}/ParamPrep/HighRes"
mkdir -p $target_dir

files=($(ls $high_res_dir/*))
in_path="${files[$SLURM_ARRAY_TASK_ID]}"
in_file="$(basename $in_path)"
echo "In path: $in_path"
out_path="${target_dir}/${in_file/.nc/}_${target_grid}_m2degr.nc"
echo "Out path: ${out_path}"

# Info on coarse graining and start

echo "Source grid path: $source_grid_path"
echo "Target grid path: $target_grid_path"
start_hcg="cdo remap,${target_grid_path},${weight_file} -sellonlatbox,$box_m2degr -setgrid,${source_grid_path} $in_path ${out_path}"
#start_hcg="remapcon,${target_grid_path} -sellonlatbox,$box_m2degr -setgrid,${source_grid_path} $in_path ${out_path}"
echo "Command: $start_hcg"

$start_hcg
