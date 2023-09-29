#!/bin/env bash

# This first line is called a "shebang" and indicates what interpreter to use
# when running this file as an executable. This is needed to be able to submit
# a text file with `sbatch`

# The necessary flags to send to sbatch can either be included when calling
# sbatch or more conveniently it can be specified in the jobscript as follows

#SBATCH -A C3SE512-22-1         # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-02:00:00          # how long time it will take to run
#SBATCH -N 1 --gpus-per-node=T4:1  # choosing no. GPUs and their type
#SBATCH -o experiment-%A_%a.out
# #SBATCH -J experiment        # the jobname (not necessary)
# #SBATCH --array 100:4

# The rest of this jobscript is handled as a usual bash script that will run
# on the primary node (in this case there is only one node) of the allocation
# Here you should make sure to run what you want to be run

# sftp

module purge
module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0
