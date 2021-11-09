#!/bin/bash
#SBATCH --ntasks=1			            	# 1 core (CPU)
#SBATCH --nodes=1			            	# Use 1 node
#SBATCH --job-name=g12_run_01	    # Name of job
#SBATCH --mem=3G 			            	# Default memory per CPU is 3GB
#SBATCH --partition=gpu                 	# Use the GPU partition
#SBATCH --gres=gpu:1                   		# Use only one GPU core
#SBATCH --mail-user=goran.sildnes.gedde-dahl@nmbu.no    		# Your email
#SBATCH --mail-type=ALL                     # Get notifications recarding your job
#SBATCH --output=outputs.out   # Output stored in this file

#===============================================================================
# This is a template for a slurm script. You need to modify this according to 
# your own experiments. You also need to choose appropriate 
# sbatch parameters above (how much memory you need is especially important).
#===============================================================================

## Script commands
module load singularity


## Define paths to relevant folders
DATADIR="<path to the dataset>"
SIFFILE="<path to the singularity container>


## Temporary results should be saved in $TMPDIR. Here is an example:
#mkdir $TMPDIR/<name-of-temporary directory>


## RUN THE PYTHON SCRIPT
# Runs a python script named run.py which takes one input argument (the data folder)
# Using a singularity container named keras.sif
singularity exec --nv $SIFFILE python run.py $DATADIR