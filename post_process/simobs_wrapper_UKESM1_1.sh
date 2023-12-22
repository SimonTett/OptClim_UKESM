#!/bin/bash
# run in serial queue unless we have changed:
# $OPTCLIMTOP/archer2/postProcess.slurm

# runs as a task int he array job under SLURM.

# currently have this script to create the environment for OPTCLIM
. /work/n02/shared/mjmn02/OptClim/setup_optclim2.sh 
# this is run in the model's directory, and has symlink history to the data.
OPT_ST_UKESM=/work/n02/shared/tetts/OptClim_UKESM/ # TODO UPDATE WHEN FINAL config ready
echo "WD is $PWD"
cmd="$OPT_ST_UKESM/post_process/comp_sim_obs_UKESM1_1.py --verbose --clean $@"
echo calling:  $cmd
result=$($cmd) # run the cmd
echo $result


