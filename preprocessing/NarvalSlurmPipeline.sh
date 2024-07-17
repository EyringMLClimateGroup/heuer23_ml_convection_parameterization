#!/bin/bash

# This file contains a preprocessing pipeline for one whole day

date
day=$1

COMMON_FLAGS="--account=bd1179 --partition=shared --parsable"
PARS_ARG="--parsable"

scratch_path="/scratch/b/b309215/HErZ-NARVALII/DATA/$day"
work_path="/work/bd1179/b309215/heuer23_ml_convection_parameterization/ProcessedAllVars/$day"
pipeline_path="/work/bd1179/b309215/heuer23_ml_convection_parameterization/preprocessing/"
log_dir="${pipeline_path}/${day}SlurmLogs"
mkdir -p $log_dir

echo "Starting job pipeline for $day ..."
## Retrieve one whole day
#jid1=$(sbatch $PARS_ARG --output=${log_dir}/slurm-retrieve_narval_recursive-%j.out ${pipeline_path}/RetrieveOneDayRecursive.sh $1)
#echo jid1: $jid1
#
## Retrieve files which where not received by last slk call
#jid2=$(sbatch $PARS_ARG --output=${log_dir}/slurm-retrieve_lost_files-%j.out --dependency=afternotok:$jid1 ${pipeline_path}/RetrieveSlkLostFiles.sh $day)
#echo jid2: $jid2

# Split first timesteps (are saved as two dts in one file as opposed to all other files)
jid3=$(sbatch $COMMON_FLAGS --output=${log_dir}/slurm-split_datasets_recursive-%j.out -t 02:00:00 --mem=16G -c 1 -N 1 -n 1 ${pipeline_path}/SplitAll0001Files.sh $scratch_path)
echo jid3: $jid3

# Create a file containing the total precip. (precip. data is saved in seperate tar file)
jid4=$(sbatch $COMMON_FLAGS --output=${log_dir}/slurm-make_tot_prec_file-%j.out --dependency=afterok:$jid3 -t 00:20:00 --mem=8G -c 4 -N 1 -n 1 ${pipeline_path}/MakeTotPrecFile.sh $scratch_path $work_path)
echo jid4: $jid4

# Create file list for files to coarse-grain / calc virt pot temp excess
jid5=$(sbatch $COMMON_FLAGS --output=${log_dir}/slurm-create_file_list-%j.out --dependency=afterok:$jid4 -t 00:10:00 --mem=1G -c 1 -N 1 -n 1 ${pipeline_path}/CreateFileList.sh $day)
echo jid5: $jid5

# Coarse virt. pot. temp. field (for environment excess)
jid6=$(sbatch $PARS_ARG --array=0-36 --output=${log_dir}/slurm-virt_temp_env-%A_%a.out --dependency=afterok:$jid5 ${pipeline_path}/VirtPotTempCoarsenRemapcon.sh DOM01 R02B08 ${pipeline_path}/in_files_${day}.txt ${work_path}/VirtPotTempRemapcon)
echo jid6: $jid6

# Calculate high res. conv output
jid7=$(sbatch $PARS_ARG --array=0-36 --output=${log_dir}/slurm-high_res_calc-%A_%a.out --dependency=afterok:$jid6:$jid4 ${pipeline_path}/SlurmCalcConvOut.sh $day $work_path)
echo jid7: $jid7

jid8=$(sbatch $PARS_ARG --array=0-36 --output=${log_dir}/slurm-ccells-%A_%a.out --dependency=afterok:$jid6:$jid4 ${pipeline_path}/SlurmCalcCcells.sh $day $work_path)
echo jid8: $jid8

# Coarse-grain horizontally to R02B04
jid9=$(sbatch $PARS_ARG --output=${log_dir}/slurm-hcg-%j.out --dependency=afterok:$jid7 ${pipeline_path}/CoarseGrainHorizontalComputeParallel.sh DOM01 R02B04 $work_path)
echo jid9: $jid9

# Coarse-grain horizontally to R02B05
jid10=$(sbatch $PARS_ARG --output=${log_dir}/slurm-hcg-%j.out --dependency=afterok:$jid7 ${pipeline_path}/CoarseGrainHorizontalComputeParallel.sh DOM01 R02B05 $work_path)
echo jid10: $jid10

# Coarse-grain horizontally to R02B06
jid11=$(sbatch $PARS_ARG --output=${log_dir}/slurm-hcg-%j.out --dependency=afterok:$jid7 ${pipeline_path}/CoarseGrainHorizontalComputeParallel.sh DOM01 R02B06 $work_path)
echo jid11: $jid11

# Coarse-grain horizontally to R02B06_0021
jid12=$(sbatch $PARS_ARG --output=${log_dir}/slurm-hcg-%j.out --dependency=afterok:$jid7 ${pipeline_path}/CoarseGrainHorizontalComputeParallel.sh DOM01 R02B06_0021 $work_path)
echo jid12: $jid12
 
# Delete scatch file to save memory
if [[ $day == "2016083200" ]]; then
    jid13=$(sbatch $COMMON_FLAGS --output=${log_dir}/slurm-del_scratch_dir-%j.out --dependency=afterok:$jid8:$jid9:$jid10:$jid11:$jid12 ${pipeline_path}/DeleteNoScratchFiles.sh $scratch_path $work_path)
else
    jid13=$(sbatch $COMMON_FLAGS --output=${log_dir}/slurm-del_scratch_dir-%j.out --dependency=afterok:$jid8:$jid9:$jid10:$jid11:$jid12 ${pipeline_path}/DeleteScratchFiles.sh $scratch_path $work_path)
fi
echo jid13: $jid13

# Coarse-grain vertically
jid14=$(sbatch $PARS_ARG --array=0-36 --output=${log_dir}/slurm-vcg-%A_%a.out --mem=4G --ntasks=1 --dependency=afterok:$jid9 ${pipeline_path}/SlurmCalcVertCoarse.sh $work_path/ParamPrep/LowRes/R02B04 R02B04)
echo jid14: $jid14

# Coarse-grain vertically
jid15=$(sbatch $PARS_ARG --array=0-36 --output=${log_dir}/slurm-vcg-%A_%a.out --mem=5G --ntasks=1 --dependency=afterok:$jid10 ${pipeline_path}/SlurmCalcVertCoarse.sh $work_path/ParamPrep/LowRes/R02B05 R02B05)
echo jid15: $jid15

# Coarse-grain vertically
jid16=$(sbatch $PARS_ARG --array=0-36 --output=${log_dir}/slurm-vcg-%A_%a.out --mem=10G --ntasks=1 --dependency=afterok:$jid12 ${pipeline_path}/SlurmCalcVertCoarse.sh $work_path/ParamPrep/LowRes/R02B06_0021 R02B06)
echo jid16: $jid16

# Coarse-grain vertically for R02B04
jid13=$(sbatch $PARS_ARG --wait --array=0-36 --output=${log_dir}/slurm-vcg-%A_%a.out --dependency=afterok:$jid10 ${pipeline_path}/SlurmRunVertcgPy.sh $work_path/ParamPrep/LowRes/R02B04 R02B04)
echo jid13: $jid13

echo "Submitted job-pipeline for $day"

#echo "Cleaning up ..."
#${pipeline_path}/DeleteScratchFiles.sh $day

#echo "Finished"
