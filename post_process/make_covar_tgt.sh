#!/usr/bin/env bash
# make covariance matrix & target values.
tgt_time='2011-01 2011-12' # 1 year for target values.
cov_time='2001-01 2020-12' # 20 year period for covariance matrix.
tgt_file="tgt_UKESM1_1_2011.json" # target values file. Will be included in the config file.
cov_file="obsErr_UKESM1_1_2001-2020.cov" # covariance matrix file. Will be included in the config file.
regions='NHX_L NHX_S T_L T_S SHX_L SHX_S NHX_L_seas' # regions we want to use.
regions_mslp='NHX_L NHX_S T_L T_S  NHX_L_seas' # MSLP is globally conserved so remove the SH extra tropics!

input_dir='P:\optclim_data\obs_data\ts_data' # where data is stored.
output_dir='C:\Users\stett2\OneDrive - University of Edinburgh\Software\OptClim_UKESM' # where to put the output files.

# cloud properties. Start so -clean!

comp_target_covariance --clean ${input_dir}/modis_cloud_ts.nc ${input_dir}/aatsr_cloud_ts.nc --regions ${regions} \
--target_time ${tgt_time} --cov_time ${cov_time} \
--tgt_file ${output_dir}/${tgt_file} --cov_file ${output_dir}/${cov_file} --verbose \
--variables Cloud_Retrieval_Fraction_Liquid Cloud_Top_Pressure Cloud_Particle_Size_Liquid

