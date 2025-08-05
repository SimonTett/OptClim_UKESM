# compute covariances and targets from two datasets
import argparse
import logging
import UKESMlib
import pandas as pd
import pathlib
import xarray as xarray
import numpy as np
#from sklearn.covariance import EmpiricalCovariance, LedoitWolf, GraphicalLassoCV
import scipy.sparse
import sklearn

import matplotlib.pyplot as plt



def seasonal_cycle(ts: xarray.Dataset, season: str) -> xarray.Dataset:
    """
    Compute the seasonal cycle for a given season.
    :param ts: xarray Dataset with time as a coordinate.
    :param season: 'jja' for June-July-August or 'djf' for December-January-February.
    :return: xarray Dataset with the seasonal cycle.
    """
    if season == 'jja':
        ts_seas = ts.where(ts['time.month'].isin([6, 7, 8]), drop=True)
    elif season == 'djf':
        ts_seas = ts.where(ts['time.month'].isin([12, 1, 2]), drop=True)
    else:
        raise ValueError("Season must be 'jja' or 'djf'.")

    resamp = ts_seas.resample(time='YS-JAN')
    mean_seas = resamp.mean().where(resamp.count() == 3)
    mean_seas = mean_seas.dropna('time').drop_dims('nbounds', errors='ignore')
    return mean_seas
def process(ts:xarray.Dataset) -> xarray.Dataset:
    # do data processing on the time series.
    resamp = ts.resample(time='YS-JAN')
    mean = resamp.mean().where(resamp.count() == 12)  # compute annual means where there are 12 months of data
    mean = mean.dropna('time').drop_dims('nbounds', errors='ignore')
    # add in NHX_land, NHX & global seasonal cycle.


    ts_delta = seasonal_cycle(ts, 'jja') - seasonal_cycle(ts, 'djf')
    rgn_names = {r:r+'_'+'seas' for r in ts_delta.coords['region'].values}
    ts_delta= ts_delta.assign_coords(region = [rgn_names.get(r,r) for r in ts.coords['region'].values])
    result = xarray.concat([mean, ts_delta], dim='region').load()
    return result

def merge_cov(covariance: pd.DataFrame, covariance2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two covariance matrixes, filling in missing values with zeros.
    This is useful when combining covariance matrices from different datasets.
    :param covariance: first covariance matrix
    :param covariance2: second covariance matrix
    :return: merged covariance matrix with missing values filled with zeros.
    """
    all_index = covariance.index.union(covariance2.index)
    all_columns = covariance.columns.union(covariance2.columns)

    cov_full = covariance.reindex(index=all_index, columns=all_columns)
    c_full = covariance2.reindex(index=all_index, columns=all_columns)

    merged = cov_full.combine_first(c_full)

    # Fill missing values
    merged: pd.DataFrame = merged.fillna(0)
    return merged
def estimate_loc_cov(data: xarray.Dataset) -> tuple[pd.Series,pd.DataFrame]:
    """
    Estimate local covariance using sklearn's GraphicalLassoCV. This tries to generate a sparse covariance matrix
    :param data: xarray Dataset with variables as data_vars and region as a coordinate.
    :return: tuple of (location, covariance) where location is a pd.Series with the estimated locations for each region
             and covariance is a pd.DataFrame with the covariance matrix.
             Both are indexed by var_name_region i.e. 'Cloud_Retrieval_Fraction_Liquid_NHX'
    """

    fn = sklearn.covariance.GraphicalLassoCV(verbose=True, mode='cd',tol=1e-2,max_iter=1000).fit
    loc=[]
    cov = None
    for name,da in data.data_vars.items():
        d = da.load().values
        result = fn(d)
        coord = [f'{name}_{r}' for r in da.coords['region'].values]
        loc += [pd.Series(result.location_, index=coord)]
        c = pd.DataFrame(result.covariance_, index=coord,columns=coord)
        if cov is None:
            cov = c
        else:
            # Merge cov and c
            cov = merge_cov(cov, c)

    loc:pd.Series = pd.concat(loc, axis=0) # concatinate all the location series into one.
    loc.index.name = 'var_name_region'
    cov.index.name = 'var_name_region'
    cov.columns.name = 'var_name_region'
    return (loc, cov)

# set up arguments,
parser = argparse.ArgumentParser(description="""Estimate eorror covariance and target values from two different osb datasets.
   Uses GraphicalLassoCV to estimate the covariance matrix and returns a target series with the mean values.  This matrix should give a sparse precision matrix.
   Outputs will be appended to existing files in the base directory.
""")
parser.add_argument('--clean', action='store_true', help='Remove existing files', default=False)
parser.add_argument("reference", help='File containing reference timeseries',type=pathlib.Path)
parser.add_argument("other", help='File containing other timeseries',type=pathlib.Path)
parser.add_argument("--target_file", help='File to write target values to as a json file', type=pathlib.Path,default='target.json')
parser.add_argument("--cov_file", help='File to write covariance matrix to as a csv file', type=pathlib.Path,default='covariance.csv')
parser.add_argument("--target_time", help='Time range to use for target values. If not provided sames years as used for cov will be used', nargs=2, type=pd.Timestamp, default=None)
parser.add_argument("--covariance_time", help='Time range to use for covariance matrix. If not provided aal years will be used', nargs=2, type=pd.Timestamp, default=None)
parser.add_argument("--regions", help='Regions to use for covariance matrix. If not provided all regions will be used',
                    nargs='+')
parser.add_argument("--variables", help='Variables to use for covariance matrix. If not provided all variables will be used',
                    nargs='+')
parser.add_argument('--verbose', action='count', help='Enable verbose logging. More --verbose the more verbose', default=0)
args = parser.parse_args()

if args.clean:
    # remove existing files if they exist
    args.target_file.unlink(missing_ok=True)
    args.cov_file.unlink(missing_ok=True)


ref = xarray.load_dataset(args.reference)
other = xarray.load_dataset(args.other)
if args.variables:
    ref = ref[args.variables]
    other = other[args.variables]



ref = process(ref)
other = process(other)

if args.regions:
    ref = ref.sel(region=args.regions)
    other = other.sel(region=args.regions)
# 1. Find common times
common_times = np.intersect1d(ref['time'].values, other['time'].values)

# 2. Select only common times
ref = ref.sel(time=common_times)
other= other.sel(time=common_times)

covariance_data = xarray.concat([ref, other], dim='obs_sample')
if args.covariance_time:
    covariance_data = covariance_data.sel(time=slice(*args.covariance_time))

covariance_data = covariance_data.stack(sample=('time', 'obs_sample')).transpose('sample', 'region')
# load the data now
covariance_data = covariance_data.load()  # force the load to avoid lazy loading issues
# estimate the covariance
tgt,cov = estimate_loc_cov(covariance_data)
if args.target_time is not None:
    # select the target time range
    tgt_sub = covariance_data.unstack().sel(time=slice(*args.target_time)).mean(['time','obs_sample'])  # mean over the sample dimension
    # and then convert to a Series based on var_name, region.
    tgt_dict = dict()
    for var_name,var_data in tgt_sub.data_vars.items():
        for region in var_data.coords['region'].values:
            tgt_dict[f'{var_name}_{region}'] = var_data.sel(region=region).mean().item()  # mean over the region
    tgt_sub = pd.Series(tgt_dict, name='target')



# save the target values to a json file
if args.target_file.exists():
    # append to existing file
    t = pd.read_json(args.target_file, typ='series')
    tgt = tgt.append(t)  # prepend the new target values to the existing ones
tgt.to_json(args.target_file, indent=4)
# save the covariance matrix to a csv file
if args.cov_file.exists():
    # append to existing file
    cov = merge_cov(pd.read_csv(args.cov_file, index_col=0), cov)
cov.to_csv(args.cov_file)
