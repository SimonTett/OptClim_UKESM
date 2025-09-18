#!/bin/bash
#SBATCH --qos=serial
#SBATCH --partition=serial
#SBATCH --account=n02-TERRAFIRMA
#SBATCH --time=00:10:00 # processing ~ 17 months takes circa 2 mins
#SBATCH --mem=16G
#SBATCH --job-name=my_simobs
#SBATCH --output=my_simobs_%j.out
#SBATCH --error=my_simobs_%j.err
#SBATCH --export=ALL
# Sadly need to set up the env on ARCHER2.
# WHich is why this script is needed.
source /work/n02/shared/tetts/OptClim_UKESM/setup_archer2 # sets up the environment
cmd="${OPT_UKESM_ROOT}/post_process/comp_sim_obs_UKESM1_1.py  $@"
echo calling: $cmd
$cmd # run the command


