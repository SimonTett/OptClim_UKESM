#!/usr/bin/env python
# compute covariances and targets from two datasets
import argparse
import logging
import typing

import UKESMlib
import pandas as pd
import pathlib
import xarray as xarray
import numpy as np
#from sklearn.covariance import EmpiricalCovariance, LedoitWolf, GraphicalLassoCV
import scipy.sparse
import sklearn

import matplotlib.pyplot as plt







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
            cov = UKESMlib.merge_cov(cov, c)

    loc:pd.Series = pd.concat(loc, axis=0) # concatinate all the location series into one.
    loc.index.name = 'var_name_region'
    cov.index.name = 'var_name_region'
    cov.columns.name = 'var_name_region'
    return (loc, cov)

# set up arguments,
parser = argparse.ArgumentParser(description="""Estimate error covariance and target values from two different osb datasets.
   Uses GraphicalLassoCV to estimate the covariance matrix and returns a target series with the mean values.  This matrix should be a sparse precision matrix.
   Outputs will be appended to existing files in the base directory.
""")
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files', default=False)
parser.add_argument('--clean', action='store_true', help='Remove the output files before doing anything else.', default=False)
parser.add_argument('--merge', action='store_true', help='Merge target and covariance into existing data. Will set --overwrite to True', default=False)
parser.add_argument("files", help='Files containing reference & other  timeseries. Provided as file_ref:file_other file_ref2:file_other2 ....',type=str, nargs='+')
parser.add_argument("--target_file", help='File to write target values to as a json file', type=pathlib.Path,default='target.json')
parser.add_argument("--cov_file", help='File to write covariance matrix to as a csv file', type=pathlib.Path,default='covariance.csv')
parser.add_argument("--target_time", help='Time range to use for target values. If not provided same years as used for cov will be used', nargs=2, type=pd.Timestamp, default=[None, None])
parser.add_argument("--covariance_time", help='Time range to use for covariance matrix. If not provided all years will be used', nargs=2, type=pd.Timestamp, default=[None, None])
parser.add_argument("--regions", help='Regions to use for covariance matrix. If not provided all regions will be used',
                    nargs='+',default=[])
parser.add_argument("--variables", type=str,help='Variables to use for covariance matrix. If not provided all variables will be used. ',
                    nargs='+',default=[])
parser.add_argument('--exclude_variables', type=str, help='Variables to exclude from the covariance matrix. If not provided no variables will be excluded.',nargs='+', default=[])
parser.add_argument('--log_level', default='WARNING', type=str, help='Log level of the script. Default is warning. Use debug or iNFO for more information.')
parser.add_argument('--mslp',nargs='+', type=str, default=[],
                    help='List of MSLP variables to process. These will be converted to difference from global mean. ')
parser.add_argument('--z', type=int, default=None,
                    help='Vertical level to use for the data. If not provided then all levels will be used. This is useful for 3D data like ERA5.')
args = parser.parse_args()

my_logger = logging.getLogger(__name__)
logging.basicConfig(level=args.log_level.upper(), format='%(asctime)s - %(levelname)s - %(message)s')

if args.merge and (not args.overwrite):
    args.overwrite = True  # if merging then we need to overwrite the target file.
    my_logger.warning('Merging target and covariance will overwrite existing files. --overwrite set to True.')




files = dict()
for file_pair in args.files:
    if ',' not in file_pair:
        raise ValueError(f"File pair {file_pair} does not contain a ',' to separate reference and other files.")
    ref_file, other_file = file_pair.split(',', 1)
    ref_file = pathlib.Path(ref_file.strip())
    other_file = pathlib.Path(other_file.strip())
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference file {ref_file} does not exist.")
    if not other_file.exists():
        raise FileNotFoundError(f"Other file {other_file} does not exist.")
    if ref_file not in files:
        files[ref_file] = other_file
    else:
        raise ValueError(f"Reference file {ref_file} already exists in the list. Each reference file must be unique.")


if args.clean:
    # remove the target and covariance files if they exist.
    if args.target_file.exists():
        my_logger.info(f'Removing existing target file {args.target_file}')
        args.target_file.unlink()
    if args.cov_file.exists():
        my_logger.info(f'Removing existing covariance file {args.cov_file}')
        args.cov_file.unlink()
if args.target_file.exists() and not args.overwrite:
    raise ValueError(f"Target file {args.target_file} already exists. Use --overwrite to overwrite it.")
if args.cov_file.exists() and not args.overwrite:
    raise ValueError(f"Covariance file {args.cov_file} already exists. Use --overwrite to overwrite it.")

exclude = set(args.exclude_variables)  # convert to give unique values
variables = set(args.variables)  # convert to set to give unique values
regions=set(args.regions)
all_tgt =[]
all_cov = pd.DataFrame() # empty dataframe to hold the covariance matrix
for ref_file, other_file in files.items():
    logging.info(f"Processing files {ref_file} and {other_file}")
    ref = xarray.open_dataset(ref_file)
    other = xarray.open_dataset(other_file)

    if variables:
        ref = ref[args.variables]
        other = other[args.variables]
    if exclude:
        ref = ref.drop_vars(exclude, errors='ignore')
        other = other.drop_vars(exclude, errors='ignore')
    logging.debug(f'Reference dataset variables: {ref.data_vars.keys()}')
    logging.debug(f'Other dataset variables: {other.data_vars.keys()}')

    # deal with vertical levels if provided
    def select_vertical_level(ds: xarray.Dataset, level: float) -> xarray.Dataset:
        """
        Select a specific vertical level from the dataset.
        :param ds: xarray Dataset to select from.
        :param level: Vertical level to select.
        :return: xarray Dataset with the selected vertical level.
        """
        _,_,levelc,_ = UKESMlib.guess_coordinate_names(ds)
        if levelc is None:
            raise ValueError(f"Dataset {ds} does not have a vertical level coordinate. Cannot select level {level}.")
        result = ds.rename({levelc: 'level'}).sel(level=level, method='nearest')
        rgn_names = [r+'@'+str(level) for r in result.coords['region'].values]
        result = result.assign_coords(region=rgn_names)
        my_logger.info(f'Extracted dataset to {result.level} at level {level}')
        result = result.drop_vars('level', errors='ignore')  # drop the level coordinate
        return result
    if args.z is not None:
        ref = select_vertical_level(ref, args.z)
        other = select_vertical_level(other, args.z)

    ref = UKESMlib.process(ref, mslp_vars=args.mslp)
    other = UKESMlib.process(other, mslp_vars=args.mslp)



    if args.regions:
        ref = ref.sel(region=args.regions)
        other = other.sel(region=args.regions)
        logging.debug(f'Reference dataset regions: {ref.coords["region"].values}')
   # now got all the data we want!
    # 1. Find common times
    common_times = np.intersect1d(ref['time'].values, other['time'].values)

    # 2. Select only common times
    ref = ref.sel(time=common_times).compute()
    other= other.sel(time=common_times).compute()
    logging.info("Processed data")

    covariance_data = xarray.concat([ref, other], dim='obs_sample')
    if args.covariance_time:
        covariance_data = covariance_data.sel(time=slice(*args.covariance_time))

    covariance_data = covariance_data.stack(sample=('time', 'obs_sample')).transpose('sample', 'region')
    # load the data now
    logging.info('Loading data now')
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
        tgt = pd.Series(tgt_dict, name='target')
    tgt.index.name = 'var_name_region'  # set the index name for the target Series
    all_tgt += [tgt]  # append the target values to the list
    all_cov = UKESMlib.merge_cov(all_cov, cov)  # merge the covariance matrix with the
    logging.info(f'Processed {ref_file} and {other_file} with {len(tgt)} target values and covariance matrix of shape {cov.shape}')
## all done so # convert the list of target values to a Series
tgt = pd.concat(all_tgt, axis=0)  # concatenate all the target values into a single Series

if args.merge:
    # load the existing target file if it exists
    if args.target_file.exists():
        existing_tgt = pd.read_json(args.target_file, typ='series')
        my_logger.debug(f'Loaded existing target values from {args.target_file}')
        tgt = pd.concat([existing_tgt, tgt], axis=0)  # merge the new target values with the existing ones
        tgt = tgt[~tgt.index.duplicated(keep='last')]  # remove duplicates, keeping the last one
        # load the existing covariance file if it exists
        if not args.cov_file.exists():
            raise FileNotFoundError(f"Covariance file {args.cov_file} does not exist. Cannot merge covariance matrix.")
        cov = pd.read_csv(args.cov_file, index_col=0)
        my_logger.debug(f'Loaded existing covariance matrix from {args.cov_file}')
        cov = UKESMlib.merge_cov(cov, all_cov)  # merge the new covariance matrix with
# clean up tgt and covariance -- removing duplicates and ensuring the index is set correctly
tgt = tgt[~tgt.index.duplicated(keep='last')]  # set the index name for the target Series
#cov = cov[~cov.index.duplicated(keep='last'),~cov.columns.duplicated(keep='last')]  # remove duplicates from the covariance matrix
cov = cov.reindex(index=tgt.index, columns=tgt.index)  # reindex the covariance matrix to match the target index
tgt.index.name = 'var_name_region'
cov.index.name = 'var_name_region'  # set the index name for the covariance DataFrame
cov.columns.name = 'var_name_region'  # set the columns name for the covariance

tgt.to_json(args.target_file, indent=2)
# save the covariance matrix to a csv file
cov.to_csv(args.cov_file)
