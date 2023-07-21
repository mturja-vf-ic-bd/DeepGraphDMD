#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=16g
#SBATCH -t 5-00:00:00

module add matlab
matlab -nodesktop -nosplash -singleCompThread -r batch_wise_gDMD -logfile batch_wise_gDMD.out