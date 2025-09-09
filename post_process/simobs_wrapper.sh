#!/bin/bash
#SBATCH --qos=serial
#SBATCH --partition=serial
#SBATCH --account=nn02-TERRAFIRMA
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --job-name=my_simobs
#SBATCH --output=my_simobs_%j.out
#SBATCH --error=my_simobs_%j.err
#SBATH --user=tetts
#SBATCH --export=ALL
#export OPT_UKESM_ROOT=/work/n02/shared/tetts/OptClim_UKESM # check this value
source "${OPT_UKESM_ROOT}"/setup_archer2 # sets up the environment
echo calling: comp_sim_obs_UKESM_atmos.py  "$@"
comp_sim_obs_UKESM_atmos.py  "$@" # then run the command

