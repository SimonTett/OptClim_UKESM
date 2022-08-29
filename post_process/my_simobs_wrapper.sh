#!/bin/bash
# run in serial queue unless we have changed:
# $OPTCLIMTOP/archer2/postProcess.slurm

# runs as a task int he array job under SLURM.

# currently have this script to create the environment for OPTCLIM
# note ~ is  not accessible from parallel job queues.

. ~/setup_optclim2.sh

# this is run in the model's directory, and has symlink history to the data.

echo calling:  $OPT_ST_UKESM/post_process/comp_sim_obs_UKESM_atmos.py  -d ./history "$@"
$OPT_ST_UKESM/post_process/comp_sim_obs_UKESM_atmos.py -d ./history  "$@"

