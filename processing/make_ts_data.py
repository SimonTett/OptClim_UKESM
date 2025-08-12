#!/usr/bin/env python
"""
Process time series data to extract relevant variables, regrid, and mask overlapping data.
Converted from make_ts_data.sh by chatgpt-4.1
"""
import logging
import os
import subprocess
import glob
import sys


import UKESMlib
import pathlib



my_logger = logging.getLogger('make_ts_data')
my_logger = UKESMlib.init_log(my_logger, level='INFO')

_search_cache = {}
os.environ['MY_SCRIPT_PATH'] = 'processing:.'
def find_file(filename: str, env_var: str = 'MY_SCRIPT_PATH') -> pathlib.Path | None:
    if filename in _search_cache:
        my_logger.debug(f'Using cached search result for {filename} = {_search_cache[filename]}')
        return _search_cache[filename]

    search_dirs = os.environ.get(env_var, '').split(':')
    for d in search_dirs:
        dir_path = pathlib.Path(d)
        candidate = dir_path / filename
        if candidate.exists():
            _search_cache[filename] = candidate
            my_logger.debug(f'Found {filename} in {dir_path}')
            return candidate

    _search_cache[filename] = None
    return None
def add_python(cmd: list) -> list:
    """
    Add 'python' to the command if not already present.
    This is useful for ensuring the command runs with Python interpreter.
    """

    if os.name.lower() == 'nt' and cmd[0] != sys.executable:
        cmd = [sys.executable] + cmd  # Add python executable at the start
        # and need to get the full path
        cmd[1] = str(find_file(cmd[1])) or cmd[1]
    return cmd

def run_command(cmd_input: list):
    """
    Run a command using subprocess.run and return the result.
    """

    cmd = [str(c) for c in cmd_input]  # Convert all elements to strings
    cmd = add_python(cmd)  # Ensure the command starts with 'python'

    my_logger.info(f'Running command: {" ".join(cmd)}')


    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        my_logger.warning(f"Command {' '.join(cmd)} failed with return code {result.returncode}")
        breakpoint()


    return result
preprocess= False
# Set environment variables
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'

base_dir = UKESMlib.base_dir
OPT_UKESM_ROOT = pathlib.Path(
    os.environ.get('OPT_UKESM_ROOT', os.getcwd())
                              )
os.environ['OPT_UKESM_ROOT'] = str(OPT_UKESM_ROOT)  # Set the environment variable for later use
data_dir = base_dir / 'obs_data'
ts_dir = UKESMlib.process_dir/'ts_data'
os.makedirs(ts_dir, exist_ok=True)

sat_cld_vars = ["Cloud_Particle_Size_Liquid","Cloud_Retrieval_Fraction_Liquid","Cloud_Top_Pressure"] # variables watned from MODIS and AATSR
sat_rename = ["Cloud_Particle_Size_Liquid:Reff", "Cloud_Retrieval_Fraction_Liquid:CLDliq","Cloud_Top_Pressure:CTP"]
cmd_root = ['comp_regional_ts.py','--land_sea_file', OPT_UKESM_ROOT/"post_process/land_frac.nc", '--nooverwrite', '--output_file']



if preprocess:
    my_logger.info('Running process_modis_aatsr.py')
    result = run_command(['process_modis_aatsr.py', data_dir])# Run process_modis_aatsr.py

    my_logger.info('Running convert_BEST.py')
    result = run_command(['convert_BEST.py', f"{data_dir}/BEST/Complete_TAVG_LatLong1.nc"]) # Run convert_BEST.py


# Modis cloud
cmd = cmd_root+ [f'{ts_dir}/modis_cloud_ts.nc', f'{data_dir}/modis_cloud_extract/*.nc']+ \
    ['--variables'] + sat_cld_vars+ ['--rename']+ sat_rename

result = run_command(cmd)

# AATSR cloud
cmd = cmd_root + [f'{ts_dir}/aatsr_cloud_ts.nc', f'{data_dir}/AATSR_cloud_extract/*.nc'] + \
    ['--variables'] + sat_cld_vars + ['--rename'] + sat_rename

result = run_command(cmd)



    


# CRU_TS temp
cmd = cmd_root+[f'{ts_dir}/CRU_TS_tmn_ts.nc', f'{data_dir}/CRU_TS/cru_ts4.09.2*.tmn.dat.nc',
                '--variables','tmn', '--rename','tmn:T2m']
result = run_command(cmd)

# CRU_TS precip
cmd = cmd_root+[f'{ts_dir}/CRU_TS_pre_ts.nc', *(data_dir/'CRU_TS').glob('cru_ts4.09.2*.pre.dat.nc'),
                '--variables','pre', '--rename','pre:Precip']
result = run_command(cmd)

# GPCC precip

cmd = cmd_root + [f"{ts_dir}/GPCC_ts.nc", *(data_dir/'GPCC').glob('full_data_monthly_v2022_2*.nc'),
       "--variables", "precip", "--rename", "precip:Precip"]
result = run_command(cmd)

# BEST temp
cmd = cmd_root + [f'{ts_dir}/BEST_ts.nc' , str(data_dir/"BEST/Complete_TAVG_LatLong1_processed.nc"),
                  '--variables', 'tmn', '--rename', "tmn:T2m"]

result = run_command(cmd)

# CERES
CERES_RENAME = "toa_lw_all_mon:OLR toa_sw_all_mon:RSR toa_net_all_mon:netflux solar_mon:INSW toa_sw_clr_c_mon:RSRC toa_lw_clr_c_mon:OLRC".split()
variables_ceres = "toa_lw_all_mon toa_sw_all_mon toa_net_all_mon solar_mon toa_sw_clr_c_mon toa_lw_clr_c_mon".split()
cmd = cmd_root + [ts_dir/"ceres_ts.nc", data_dir/"ceres/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202504.nc"]+[
    "--rename",*CERES_RENAME, '--variables', *variables_ceres]
result = run_command(cmd)
# AATSR TOA radiances
variables = "toa_swup toa_swdn toa_swup_clr toa_lwup toa_lwup_clr".split()
rename = "toa_swup:RSR toa_swdn:INSW toa_swup_clr:RSRC toa_lwup:OLR toa_lwup_clr:OLRC".split()
cmd = cmd_root + [ts_dir/'aatsr_toa_flux_ts.nc', data_dir/"AATSR_cloud/*ESACCI-L3C_CLOUD-CLD_PRODUCTS-AATSR_ENVISAT-fv3.0.nc"] + \
    ['--variables', *variables, '--rename', *rename]



result = run_command(cmd)

# ERA5
rename_era = "t2m:T2m t:T tp:Precip msl:MSLP  r:RH".split()
variables_era = "t2m t tp msl r".split()  # variables to extract from ERA5
era5_files = list((data_dir/'ERA5').glob("*.nc"))
for f in era5_files:
    out_file = f.stem
    out_file = ts_dir/(out_file+"_ts.nc")
    print(f"{f} -> {out_file}")
    cmd = cmd_root + [ out_file, str(f), '--rename', *rename_era,'--variables', *variables_era,'--time_range', '1980-01', '2024-12']
    result = run_command(cmd)

# ncep2
rename_ncep2 = "air:T mslp:MSLP  rhum:RH".split()
variables_ncep2 = "air mslp rhum".split()  # variables to extract from NCEP2
ncep_files = list((data_dir/'ncep2').glob("*.nc"))
for f in ncep_files:
    out_file = f.stem
    out_file = ts_dir/("ncep2_"+out_file+"_ts.nc")
    print(f"{f} -> {out_file}")
    cmd = cmd_root + [ out_file, str(f), '--rename', *rename_ncep2,'--variables', *variables_ncep2]
    result = run_command(cmd)