#!/usr/bin/env python
"""
Process a set of input files to compute target and covariance values.
This script processes input files to compute target and covariance values,
and saves the results to output files.

Handles specials -- netflux_global where uncertainty is 0.1 W/m^2 (See Loeb et al, 2018).
"""
import UKESMlib
import pandas as pd
import pathlib
import json
import numpy as np
import xarray


target_file = pathlib.Path('configs/target_2011.ijson')
cov_file = pathlib.Path('covariance/Obserr_covariance_2011.csv')
scaling_file = pathlib.Path('configs/scalings.ijson')
obslist_file = pathlib.Path('configs/obsList.ijson')
tgt_time = ['2011-01', '2011-12']  # target time period
base_cmd = ['processing/comp_target_covariance.py','--merge',
            '--target_file',target_file,
            '--cov_file',cov_file,
            '--target_time',*tgt_time,
            '--covariance_time','2001-01','2022-12',
            '--log_level','INFO'
            ]
rgns = ['--regions',  'NHX', 'T', 'SHX', 'NHX_L_seas']
# TOA fluxes. CERES vs AATSR
cmd = base_cmd + [str(UKESMlib.process_dir)+'/ts_data/ceres_ts.nc'+','+str(UKESMlib.process_dir)+'/ts_data/aatsr_toa_flux_ts.nc',
                  '--clean','--exclude','netflux', 'INSW',*rgns]
result = UKESMlib.run_command(cmd)
# AATSR cloud vs MODIS cloud
file = str(UKESMlib.process_dir)+'/ts_data/aatsr_cloud_ts.nc,'+str(UKESMlib.process_dir)+'/ts_data/modis_cloud_ts.nc'
cmd = base_cmd + [file, *rgns]
result = UKESMlib.run_command(cmd)

# RH & T at 500 hPa.
regns_500 = ['--regions', 'NHX@500', 'T@500', 'SHX@500','NHX@500_seas']
file = str(UKESMlib.process_dir)+'/ts_data/ERA5_RH_T_ts.nc,'+str(UKESMlib.process_dir)+'/ts_data/ncep2_rhum.mon.mean_ts.nc'
cmd = base_cmd + [file, '--variables', 'RH',  *regns_500,'--z', '500']
result = UKESMlib.run_command(cmd)

file = str(UKESMlib.process_dir)+'/ts_data/ERA5_RH_T_ts.nc,'+str(UKESMlib.process_dir)+'/ts_data/ncep2_air.mon.mean_ts.nc'
cmd = base_cmd + [file, '--variables', 'T',  *regns_500,'--z', '500']
result = UKESMlib.run_command(cmd)

# Precip & Temp -- just land. Drop the SHX as not much land there.
rgns_land = ['--regions', 'NHX_L', 'T_L', 'NHX_L_seas']
# precip CRU vs GPCC
cmd = base_cmd + [str(UKESMlib.process_dir)+'/ts_data/CRU_TS_pre_ts.nc'+','+str(UKESMlib.process_dir)+'/ts_data/GPCC_ts.nc',
                  *rgns_land]
result = UKESMlib.run_command(cmd)

# temp CRU vs BEST
files = str(UKESMlib.process_dir)+'/ts_data/CRU_TS_tmn_ts.nc'+','+str(UKESMlib.process_dir)+'/ts_data/BEST_ts.nc'
cmd = base_cmd + [files, *rgns_land]
result = UKESMlib.run_command(cmd)

# SLP. Want NHX-global & T-global & NHX_seas
file = (str(UKESMlib.process_dir)+'/ts_data/ERA5_sfc_t_mslp_1979_2024_ts.nc,'+
        str(UKESMlib.process_dir)+'/ts_data/ncep2_mslp.mon.mean_ts.nc')
rgns_mslp = ['--regions', 'NHX_DGM', 'T_DGM', 'NHX_DGM_seas']
cmd = base_cmd + [file, '--variables', 'MSLP', *rgns_mslp,'--mslp','MSLP']
result = UKESMlib.run_command(cmd)
##
final_target = pd.read_json(target_file, typ='series')
final_cov = pd.read_csv(cov_file, index_col=0)
## add in specials: netflux_global where uncertainty is 0.1 W/m^2 (See Loeb et al, 2018).
with xarray.set_options(keep_attrs=True):
    nf = xarray.open_dataset(UKESMlib.process_dir/'ts_data/ceres_ts.nc').\
        sel(region='global',time=slice(*tgt_time)).netflux.mean('time')

name = 'netflux_global'
final_target[name] = float(nf)  # convert to scalar
netflux_cov = pd.DataFrame(0.1**2, index=[name], columns=[name])  # uncertainty is 0.1 W/m^2
final_cov = UKESMlib.merge_cov(final_cov, netflux_cov)
# write the final target and covariance to files.
final_target.to_json(target_file, indent=2)
final_cov.to_csv(cov_file, index_label='region')

# set scaling factors.
scaling= dict(_comment='Scaling. MSLP -> hPa, Precip -> mm/day, Reff -> microns, CLD -> percent')
for rgn in final_target.index:
    if rgn.startswith('MSLP_') :
        scaling[rgn] = 1e-2 # convert values to hPa
    elif rgn.startswith('Precip'):
        scaling[rgn] = 24*60*60 # convert from kg/m^2/s to mm/day
    elif rgn.startswith('Reff_'):
        scaling[rgn] = 1e6 # conver to microns
    elif rgn.startswith('CLDliq_') or rgn.startswith('CLDice_'):
        scaling[rgn] = 1e2  # convert to percent
    else:
        pass
with scaling_file.open('wt') as fp:
    json.dump(scaling, fp, indent=2)
scaling.pop('_comment')



scalings = pd.Series(scaling, name='scaling',dtype=float)
# fill with ones
scalings = scalings.reindex(final_target.index, fill_value=1.0)
scaled_target = final_target * scalings
cov_scale = pd.DataFrame(np.outer(scalings, scalings), index=scalings.index, columns=scalings.index).astype(float)
scaled_cov = final_cov * cov_scale






