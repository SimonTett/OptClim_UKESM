#!/usr/bin/env python

"""
Python script to make obs targets and error covariance matrix... 

Data sources
1) CERES -- netflux, RSR, OLR, RSRC,OLRC -- internal error
2) CRU_TS -- Land Precip  -- error ~5% from Rodell et al, -- doi:10.1175/JCLI-D-14-00555.1
3) CRU_TS -- Land Temp --error vs BEST
4) ERA-5 Temp@500 hPa, RH@500 hPa & MSLP. Error vs NCEP2
5) MIDAS -- Reff -- internal error
"""

import json
import pathlib
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description="""
Extract data from processed observations and generate target obs data. 
Very specific to obs data actually used
1) CERES -- netflux, RSR, OLR, RSRC,OLRC -- internal error
2) CRU_TS -- Land Precip  -- error vs GPCC
3) CRU_TS Land Temp -- error vs BEST 
4) ERA-5 Temp@500 hPa, RH@500 hPa & MSLP. Error vs NCEP2
5) MIDAS -- Reff -- internal error

Names will be changed to "standard" names

example use:
make_obs_targets_covar.py obs_data.json target.json obs_error_cov.csv
""")

parser.add_argument("INPUT", help="The Name of the Input JSON file")
parser.add_argument("TARGET", help="The Name of the  JSON file where the target information is to be written")
parser.add_argument("OBSERRCOV", help='Name of the obs error covariance .csv file to write')
parser.add_argument("-v", "--verbose", help="Provide verbose output", action="count", default=0)
args = parser.parse_args()  # and parse the arguments
verbose = args.verbose

with open(args.INPUT, 'r') as fp:
    input_data = json.load(fp)

# vars for the file names.. Hard wired!

CERES = 'CERES/CERES_EBAF-TOA_Ed4.1_Subset_200003-202105_N48.nc'
CRU_TS_precip = 'CRU_TS/cru_ts4.05.2011.2020.pre.dat_N48_land.nc'
CRU_TS_temp = 'CRU_TS/cru_ts4.05.2011.2020.tmp.dat_N48_land.nc'# VN 4.05 temp looks suspect
ERA5_free_atmos = 'ERA5/ERA5_500hPa_RH_T_mon_2010_2019_N48.nc'
ERA5_sfc = 'ERA5/ERA5_sfc_mon_2010_2020_N48.nc'
ERA5_sfc_land = 'ERA5/ERA5_sfc_mon_2010_2020_N48_land.nc'
NCEP_mslp = 'NCEP2/mslp.mon.mean_N48.nc'
NCEP_rh = 'NCEP2/rhum_500.mon.mean_N48.nc'
NCEP_T = 'NCEP2/air_500.mon.mean_N48.nc'
MODIS = 'MODIS/MODIS_AQUA_Reff_N48.nc'
GPCC = 'GPCC/full_data_monthly_v2020_2011_2019_025_N48_land.nc'
BEST = 'BEST/Complete_TAVG_LatLong1_fix_N48.nc'
rewrite_rules = dict()  # rewrite rules -- only these ones will be used!
rewrite_rules[CERES] = dict(toa_sw_all_mon='RSR', toa_lw_all_mon='OLR',
                            toa_net_all_mon='netflux',
                            toa_sw_clr_c_mon='RSRC', toa_lw_clr_c_mon='LWC',
                            )
rewrite_rules[CRU_TS_precip] = dict(pre='Lprecip')
rewrite_rules[GPCC] = dict(precip='extra_Lprecip')
rewrite_rules[CRU_TS_temp]=dict(tmp='LAT')
rewrite_rules[BEST] = dict(abs_temp='extra_LAT')
rewrite_rules[ERA5_free_atmos] = dict(r='RH@500', t='TEMP@500')
rewrite_rules[NCEP_rh] = dict(rhum='extra_RH@500')
rewrite_rules[NCEP_T] = dict(air='extra_TEMP@500')
rewrite_rules[ERA5_sfc] = dict(msl='MSLP')
#rewrite_rules[ERA5_sfc_land] = dict(t2m='extra_LAT')
rewrite_rules[NCEP_mslp] = dict(mslp='extra_MSLP')
rewrite_rules[MODIS] = dict(Cloud_Effective_Radius_Liquid='Reff')
# create the target dict
target = dict()
for file, rewrite in rewrite_rules.items():  # iterate over the files
    data = input_data[file]  # data we want for this file.
    wanted = np.array(list(rewrite.keys()))  # what we are rewritting
    newName = np.array(list(rewrite.values()))  # new name
    for k, v in data.items():  # iterate over the rewrite rules for this file
        name, suffix = k.rsplit("_", maxsplit=1)  # split it up (last bit is GLOBAL, NHX etc)
        if suffix.endswith('DGM'): # fix the DGM
            suffix = suffix[0:-3]+"_"+suffix[-3:]
        L = (name == wanted)  # True if match/F if not
        if L.sum() > 0:  # got it.
            outkey = newName[L][0] + "_" + suffix  # new name
            target[outkey] = v

# remove all GLOBAL *except* netflux
global_values = {}
keys = list(target.keys())
for k in keys:
    if (k.split('_')[-1] == 'GLOBAL') and (k != 'netflux_GLOBAL'):
        global_values[k] = target.pop(k)  # get rid of globals except for netflux
    elif (k.split("_")[0] == 'netflux') and (k.split("_")[1] != 'GLOBAL'):
        target.pop(k)  # get rid of netflux_XXwhen XX is not global

# TODO generate errors.
# For CERES values use literature --see most recent paper. 
# For LPrecip -- abs(delta) vs GPCC
# For LAT -- abs(delta) vs ERA-5 ?? 
# For TEMP@500 -- abs(delta) vs NCEP2
# For RH@500 -- abs(delta) vs NCEP2
# For MSLP -- abs(delta) vs NCEP2
# FOr Reff ?? 
delta = {}
#loop over non extra keys in the target
for key in [k for k in target.keys() if not k.startswith('extra_')]:
    if key.startswith('Lprecip_'):
        # bigger of difference or  5% error from Rodell et al, 2015
        delta[key] = np.max([target[key]*0.05, target.pop('extra_'+key)])
    elif key.startswith('Reff_'):
        delta[key] = target[key] * 0.1
        # assume 10% error see subsection V-C-1 of Minnis et al, 2021 DOI: 10.1109/TGRS.2020.3008866
    elif key == 'netflux_GLOBAL':
        delta[key] = 0.25 # from Tett et al 2013! Wonder if there is a better est.?
    # values below come from Loeb et al, 2018 doi:10.1175/JCLI-D-17-0208.1
    # Note these are larger than Tett et al 2013 & 2018.
    elif key.startswith('RSR_'):
        delta[key] = 2.5
    elif key.startswith('OLR_'):
        delta[key] = 2.5
    elif key.startswith('SWC_'):
        delta[key] = 5.
    elif key.startswith('LWC_'):
        delta[key] = 4.5
    else:
        pass

# loop over any extras we have and deal with them
keys = [ key for key in target.keys() if key.startswith('extra_')]
for key in keys:
    key_want = key.replace('extra_', '')
    delta[key_want] = np.abs(target[key_want] - target.pop(key))


with open(args.TARGET, 'w') as fp:
    json.dump(target, fp, indent=2)


def set_corr_block(corr, var=None):
    """

    :param corr -- correlation matrix
    :param var -- default None. name or list of of variables to set (i.e. "RSR"  or
         ["RSR","OLR"]). If None then vars set to unique elements of index (to left of _)
    :return -- returns modified correlation.
    """

    if var is None:
        vars = set([s[0] for s in corr.index.str.split("_")])
    elif isinstance(var, str):
        vars = [var]
    else:
        vars = var
    mod_corr = corr.copy()  # copy the input!

    for name in vars:
        L = corr.index.str.startswith(name + '_')
        mod_corr.loc[L, L] = 1.0  # simple model correlation is 1.

    return mod_corr


ser = pd.Series(delta)
cov = pd.DataFrame(np.outer(ser, ser), index=ser.index, columns=ser.index)
corr = pd.DataFrame(np.identity(len(ser)), index=ser.index, columns=ser.index)
corr = set_corr_block(corr)
cov *= corr
cov.to_csv(args.OBSERRCOV)
