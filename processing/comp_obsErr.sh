#!/usr/bin/env bash
# compute ObsError covariance matrix
export PYTHONUNBUFFERED=1 # no buffering for python.
export PYTHONWARNINGS=ignore # turn off a bunch of warnings...
data_dir=${BASE_DIR}/obs_data
ts_dir=${data_dir}/ts_data # where the timeseries data is stored.
if [[ -z ${OPT_UKESM_ROOT} ]]
then
    echo "OPT_UKESM_ROOT is not set. Exiting"
    exit 1
fi
echo "OPT_UKESM_ROOT: ${OPT_UKESM_ROOT}"
##sw_dir=~/tetts/OptClim_UKESM/
##export PATH=$PATH:${sw_dir}/processing:${sw_dir}/post_process
# activate the environment before running the script.
echo  $VIRTUAL_ENV

tgt_file=$1; shift # get the file for the target values
covariance_file=$1 ; shift # get the file for the covariance matrix

