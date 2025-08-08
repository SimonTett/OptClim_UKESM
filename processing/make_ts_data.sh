#!/usr/bin/env bash
# process obs data to timeseries.
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
mkdir -p $ts_dir

cmd_root="comp_regional_ts.py  --land_sea_file ${OPT_UKESM_ROOT}/post_process/land_frac.nc --nooverwrite --output_file "

# run process_modis_aatsr.py to process the modis and aatsr data.
process_modis_aatsr.py ${data_dir}
# and to fix problems with BEST data
convert_BEST.py ${data_dir}/BEST/Complete_TAVG_LatLong1.nc

## modis cloud
cmd="${cmd_root} ${ts_dir}/modis_cloud_ts.nc ${data_dir}/modis_cloud_extract/*.nc"
echo "Processing modis cloud with $cmd"
$($cmd) # modis cloud

# AATRSR cloud
cmd="${cmd_root} ${ts_dir}/aatsr_cloud_ts.nc ${data_dir}/AATSR_cloud_extract/*.nc"
echo "Processing AATSR cloud data with $cmd"
$($cmd) # run it.

# CRU_TS temp
cmd="${cmd_root} ${ts_dir}/CRU_TS_tmn_ts.nc ${data_dir}/CRU_TS/cru_ts4.09.2*.tmn.dat.nc --variables tmn --rename tmn:T2m"
echo "Processing CRUTS tmn with $cmd"
$($cmd) # CRU_TS

# CRU_TS precip
cmd="${cmd_root} ${ts_dir}/CRU_TS_pre_ts.nc ${data_dir}/CRU_TS/cru_ts4.09.2*.pre.dat.nc --variables pre --rename pre:Precip"
echo "Processing CRUTS pre with $cmd"
$($cmd) # CRU_TS


# GPCC precip
cmd="${cmd_root} ${ts_dir}/GPCC_ts.nc ${data_dir}/GPCC/full_data*.nc --variables precip --rename precip:Precip"
echo "Processing CRUTS pre with $cmd"
$($cmd) # CRU_TS

# BEST temp
cmd="${cmd_root} ${ts_dir}/BEST_ts.nc ${data_dir}/BEST/Complete_TAVG_LatLong1_processed.nc --rename absolute_temperature:T2m" # compatibility with CRU_TS data/
echo "Processing BEST with $cmd"
$($cmd) # BEST

# CERES
cmd="${cmd_root} ${ts_dir}/ceres_ts.nc ${data_dir}/ceres/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202504.nc"
echo "Processing CERES with $cmd"
$($cmd) # CERES


# ERA5
rename_era="t2m:T2m r:T tp:Precip msl:MSLP  r:RH"
era5_files=$(ls -1  ${data_dir}/ERA5/*.nc)
for f in ${era5_files}
do
    out_file="$(basename $f .nc)"
    out_file="${ts_dir}/${out_file}_ts.nc"
    echo "$f -> $out_file"
    cmd="${cmd_root} ${out_file} ${f}"
    echo "Processing ERA5 file ${f} with $cmd"
    $($cmd) # ERA-5
done
# END ERA5
    
# ncep2
ncep_files=$(ls -1  ${data_dir}/ncep2/*.nc)
rename_ncep2="air:T mslp:MSLP  rhum:RH"
for f in ${ncep_files}
do
    out_file="ncep2_$(basename $f .nc)"
    out_file="${ts_dir}/${out_file}_ts.nc"
    echo "$f -> $out_file"
    cmd="${cmd_root} ${out_file} ${f} --rename ${rename_ncep2}"
    echo "Processing ncep2 file ${f} with $cmd"
    $($cmd) # 
done
# END ncep2





