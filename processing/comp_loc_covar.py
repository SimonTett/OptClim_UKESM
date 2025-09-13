#!/usr/bin/env python
"""
Compute location and covariance values from multiple timeseries files.
"""
import argparse
import datetime
import sys
import typing
from collections import defaultdict

import UKESMlib
import pandas as pd
import pathlib
import xarray as xarray
import numpy as np
import sklearn
import scipy.sparse
import json
import re


my_logger=UKESMlib.my_logger
def split_str(s):
    """
    Split a string into base and suffix where suffix is either _STR or _STR_seas
    :param s: input string
    :return: tuple of base string and suffix (including leading _)
    """
    # Match: any string, then _STR or _STR_seas at the end
    m = re.match(r'^(.*?)(_([A-Za-z0-9]+)(_seas)?)$', s)
    if m:
        return m.group(1), m.group(2)
    else:
        return s, ''

def split_str(s:str, patterns:list[str]) -> tuple[str,str]:
    """
    Split a string into base and suffix where suffix is from list of patterns
    :param s: input string
    :param patterns: list of suffix patterns to match
    :return: tuple of base string and suffix (excluding  _)
    """
    # Escape and join patterns for regex
    pattern_group = '|'.join(re.escape(p) for p in patterns)
    # Build regex: match anything, then one of the patterns at the end
    regex = rf'^(.*)_({pattern_group})$'
    m = re.match(regex, s)
    if m:
        return m.group(1), m.group(2)
    else:
        return s, ''
def set_option(defaults:dict[str,typing.Any], parser:argparse.ArgumentParser) -> dict[str, typing.Any]:
    """
    Set the options from the (in high to low priority order) argparse namespace, config file and defaults.
    options will contain the values with keys given by the args names and an additional key <arg_name>_help
    containing the help string for that argument.  Values of null in the config file will overwrite defaults while values of None in args
    will not overwrite anything.
    """
    args = parser.parse_args()

    options = defaults.copy()
    if args.config:
        with args.config.open('rt') as fp:
            json_options = json.load(fp)  # load the options from the config file
        options.update(json_options.items())

    # overwrite the options with arg values if they were set
    options.update({
        arg_name: value for arg_name, value in vars(args).items() if not (value is None or value ==[]) })

    # set types and help info appropriately using agrs. Makes use of *PRIVATE* attributes of argparse.
    for act in parser._actions:  # hack using private part of parser
        if act.type is not None:
            try:
                if isinstance(options[act.dest], list):
                    options[act.dest] = [act.type(v) for v in options[act.dest]]
                else:
                    options[act.dest] = act.type(options[act.dest])
            except KeyError:
                my_logger.debug(f'Did not find {act.dest} so no type conversion')
            except TypeError:
                my_logger.warning(f'Problem converting {act.dest} to type {act.type}. Leaving as {type(options[act.dest])}')
        if act.help:
            options[act.dest+'_help'] = act.help # store the help string for this option

    return options

def sort_by_var(var_region:list[str],ordering:list[str]) -> list[str]:
    """
    Sort a list of variable_region strings by the variable name according to the ordering list.
    The variable name is assumed to be the part of the string before the last underscore.
    Any variable not in the ordering list will be placed at the end in original order.
    :param var_region: Name of variable regions. Everything after last _ is assumed to be region; everything before is variable name
    :param ordering: List of variable names in desired order
    :return: ordered list.
    """

    def strip_suffix(s):
        return s.rsplit("_", 1)[0]

    # Create a mapping from stripped name to its order index
    order_map = {name: idx for idx, name in enumerate(ordering)}

    # Sort variables by their stripped name's position in order
    sorted_vars = sorted(
        var_region,
        key=lambda v: order_map.get(strip_suffix(v), float('inf'))
    )

    return sorted_vars

def process(data_array:xarray.DataArray) \
        -> typing.Optional[tuple[xarray.DataArray, pd.Series]]:
    """
    Process  the data array  and work out scaling
    :param data_array: xarray DataArray to process
    :return: processed data array and pandas series of scalings.
    """
    var_name = data_array.name
    if (options.get('rename') is not None ) and var_name in options['rename']:
        var_name = options['rename'][var_name]
        data_array = data_array.rename(var_name)
        my_logger.debug(f'Renamed {data_array.name} to {var_name}')

    if options.get('levels',{}).get(var_name) is not None:
        # FIXME -- for now assumed to be a single level. If need multiple levels will need to loop over them
        # that will break code below that assumes a single data array per variable :-(
        _,_,vertc,timec = UKESMlib.guess_coordinate_names(data_array)
        level =options['levels'][var_name]
        if isinstance(level,list):
            raise NotImplementedError('Multiple levels not implemented yet')

        coord= {vertc: level}
        data_array = data_array.sel(**coord).rename(var_name+'@'+str(level))
        my_logger.debug(f'Extracted {vertc}= {level} for variable {var_name} renaming to {data_array.name}')
        var_name = data_array.name

    if (var_name in options.get('exclude_variables',[])  or  # in exclude variables ?
            (options.get('variables') is not None and # variables is not None and var_name not in the list.
            var_name not in options['variables'])):
        my_logger.debug(f'Excluding variable {var_name}')
        return #
    # Optionally process the data
    if options.get('process',True):
        if var_name in options.get('mslp_vars',[]):
            my_logger.debug(f'Processing MSLP variable {var_name}')
            data_array  = UKESMlib.mslp_process(data_array)
            # change name to indicate delta from global mean
            var_name = var_name+'_DGM'
        data_array = UKESMlib.process(data_array)  # process the data if needed
        my_logger.debug(f'Processed data for {var_name}')
    # extract regions if needed
    if var_name in options.get('special_regions',[]) :
        # special regions defined and var_name in special regions
        data_array = data_array.sel(region=options['special_regions'][var_name])
        my_logger.debug(f'Using special regions for {var_name}: {options["special_regions"][var_name]}')
    elif options.get('regions') is not None: # got some regions to use
        data_array = data_array.sel(region=options['regions'])
        my_logger.debug(f'Using regions for {var_name}: {options["regions"]}')
    else:
        pass # nothing to do!


    # work out scaling for this variable
    scales={f'{var_name}_{regn}':options['scales'].get(var_name,1.0) for regn in data_array.region.values}
    scales=pd.Series(scales).rename(var_name)

    # work out the realisation and add to the data array.
    indx = realisation_count[var_name]
    realisation_count[var_name]+=1
    data_array = data_array.assign_coords(realisation=indx).rename(var_name)

    return data_array,scales
# Possible covariance estimators
covariance_estimators = dict(
    LedoitWolf = sklearn.covariance.LedoitWolf(assume_centered=True), #
    GraphicalLasso = sklearn.covariance.GraphicalLasso(assume_centered=True), # Lasso -- spare precision matrices
    EmpiricalCovariance = sklearn.covariance.EmpiricalCovariance(assume_centered=True) # empirical covariance for comparison
)
covariance_estimators_meta= dict(
    LedoitWolf='shrinkage_', # report on the shrinkage
    GraphicalLasso='n_iter_' # report on the number of iterations
)

def get_metadata(cov_object:typing.Any,estimator_name:typing.Optional[str],) -> typing.Optional[tuple[str,float]]:
    """
    Get the metadata attribute name and value for the given estimator name.
    :param cov_object: The covariance estimator object.
    :param estimator_name: The name of the  estimator.
    """
    if estimator_name is None:
        return None
    attr = covariance_estimators_meta.get(estimator_name)
    try:
        value = float(getattr(cov_object,attr))
        return attr, value
    except (TypeError, AttributeError):
        return None

defaults = dict(location_file='location.json',
                cov_file='covariance.csv',
                log_level='WARNING',
                overwrite=False,
                block=False,
                print_default=False,
                process=True, # whether to process the data or not. Data might have already been processed...
                covariance_estimator = 'LedoitWolf', #  covariance estimator to use
                ) # default values which are not None.
# provide default here so can be overridden by config file.
# DO not provide defaults in args. If nothing set will be assumed None
parser = argparse.ArgumentParser(description="""Estimate  covariance and mean values from at least two time series 
   'location' and covariance matrix are written out as pandas series and dataframe to json and csv file. 
     Arguments can also be provided in a json config file using --config option. Command line options override config file values which override defaults.""",
                                 epilog="--rename, --special_regions,z_levels, --override_cov &  --scales all use json.loads to decode. Easier to pass in via the config file.")

parser.add_argument("files", help='List of files to read in. Must glob to at least two files.',type=UKESMlib.expand, nargs='*',default=None)
# default = None so we don't get an empty list if nothing provided
parser.add_argument("--config",
                    help="Config file path whose values will overwrite default values but be overwritten by command line values", type=UKESMlib.expand)# file arguments
# file output args
parser.add_argument("--cov_file", help='File to write covariance matrix to as a csv file', type=UKESMlib.expand)
parser.add_argument("--location_file", help='File to write mean values to as a json file', type=UKESMlib.expand)
parser.add_argument("--metadata_file", help='File to write metadata to as a json file', type=UKESMlib.expand)
parser.add_argument("--scalings_file", help='JSON file where scalings to be written to', type=UKESMlib.expand)
parser.add_argument("--variables_file", help='JSON file where variables used are to be written to', type=UKESMlib.expand)
## options for time
parser.add_argument("--cov_time",
                    help='Time range to use for covariance matrix. If not provided all years will be used', nargs=2)
parser.add_argument("--location_time",
                    help='Time range to use for location values. If not provided same years as used for cov will be used',
                    nargs=2)
## options for variables and regions
parser.add_argument("--rename",type=json.loads, help='A json dictionary of variable names to rename. E.g. \'{"T2m":"T"}\'')
parser.add_argument("--variables",
                    help='Variables to use for covariance matrix. If not provided all common variables will be used. ',
                    nargs='+')
parser.add_argument('--exclude_variables',
                    help='Variables to exclude from the calculations. If not provided no variables will be excluded.',
                    nargs='+')
parser.add_argument('--regions', nargs='+',
                    help='List of regions to process. If not provided all regions will be used.')

parser.add_argument('--special_regions', type=json.loads,
                    help='A json dictionary of variable names to list of regions to use for those variables. E.g. \'{"T2m":["NHX_L","T_L"]}\'')
parser.add_argument('--exclude_vars_files',type=json.loads,
                    help="a json dict indexxed by variable with list of file *names* to exclude.")
parser.add_argument('--mslp_vars', nargs='+',
                    help='List of variables that are mean sea level pressure. These will be converted to delta from global mean values.')
parser.add_argument('--levels',type=json.loads,help='A json dictionary of variable names to levels to use for those variables. E.g. \'{"T":500}\'')
parser.add_argument('--override_cov',type=json.loads,
                    help='A json dictionary of variable_regions to override/supplement the covariances values E.g. \'{"netflux_global":1e-2}\'. Off diagonal values will be zeroed')
parser.add_argument('--override_loc',type=json.loads,
                    help='A json dictionary of variable_regions to override/supplement the location values E.g. \'{"Cess":2.0}\' ')
## other options
parser.add_argument('--covariance_estimator', choices=covariance_estimators.keys(),help='Covariance estimator to use')
parser.add_argument('--block',action=argparse.BooleanOptionalAction,
                    help='Each variables is considered independent with only covariances within a variable calculated.')
parser.add_argument('--process', action=argparse.BooleanOptionalAction,
                    help='Whether to process the data or not.')
parser.add_argument('--scales', type=json.loads,
                    help='A json dictionary of variable names to scale the variables by. E.g. \'{"Precip":86400}\'')
parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, help='Overwrite existing files')
parser.add_argument('--log_level', help='Log level of the script. ')
parser.add_argument('--print_default', action=argparse.BooleanOptionalAction,
                    help='Print the default options and exit')


options = set_option(defaults, parser)
# setup processing
UKESMlib.setup_logging(options['log_level'])
# now all setup.
# optionally show defaults and exit
if options.get('print_default', False):
    print("Default options are:")
    for k, v in defaults.items():
        print(f"  {k}: {v}")
    sys.exit(0)

# Show options
# write out the options to logger
for k,v in options.items():
    my_logger.debug(f'Option[{k}] = {v}')

# test if output_files exist and overwrite not set. Then  create directories if needed
for k,f in options.items():
    if not isinstance(f,pathlib.Path):
        continue
    if (not options['overwrite']) and f.exists():
        raise FileExistsError(f'Output file {f} already exists. Use --overwrite to overwrite it.')
    else:
        f.parent.mkdir(parents=True, exist_ok=True) # make the directory if it does not exist



# read in the datasets
files=[]
for file in options['files']: # handle globs. Really for %run in ipython on windows
    gfiles = sorted(file.parent.glob(file.name)) # list of files
    if len(gfiles) == 0:
        my_logger.warning(f'No files found matching {file}')
    files += gfiles
    my_logger.debug(f'Found {len(gfiles)} files matching {file}')
my_logger.info(f'Found {len(files)} files')


excluded_file_vars=defaultdict(list) # dict of excluded files indexed by variable
timeseries=defaultdict(list) # contains the timeseries read in from each file
realisation_count = defaultdict(int) # count of realisations for each variable
scale_units={} # scalings for data. Keyed by variable name
for file in files:
    my_logger.info(f'Reading file {file}')
    with xarray.open_dataset(file) as ds:
        for var in ds.data_vars:
            try:
                file_names = options['exclude_vars_files'][var]

                if any([file_name == file.name for file_name in file_names]):
                    # iterate over file_names
                    my_logger.info(f'Skipping variable {var} in file {file}')
                    excluded_file_vars[var].append(str(file))
                    continue # onto the next variable in the file
            except KeyError: # either no exclude_vars_file or var in it.
                pass
            process_result = process(ds[var])
            if process_result is None:
                continue
            da,scale= process_result
            timeseries[da.name].append(da)
            scale_units[scale.name]=scale
# now have all the data read in. Need to combine the realisations for each variable
mn_timeseries = dict() # where the mean timeseries go
slice_cov_time = slice(*options['cov_time']) if options.get('cov_time') is not None else None
slice_location_time = slice(*options['location_time']) if options.get('location_time') is not None else slice_cov_time
vars_to_keep = list(timeseries.keys())

for var in timeseries.keys():
    ts = xarray.concat(timeseries[var], dim='realisation', join='outer') # merge the realisations for this variable
    msk = np.isnan(ts).any('realisation')
    # remove anything that is nan in any realisation
    ts = ts.where(~msk, drop=True)
    mn_ts = ts.mean(dim='realisation') # mean across realisations
    ts = ts - mn_ts  # remove the mean across realisations
    if slice_cov_time is not None:
        ts = ts.sel(time=slice_cov_time) # select the time
    if slice_location_time is not None:
        mn_ts = mn_ts.sel(time=slice_location_time) # select the time


    mn_timeseries[var] = mn_ts
    timeseries[var] = ts
    my_logger.info(f'Variable {var} after processing has shape {timeseries[var].shape} and mean shape {mn_timeseries[var].shape}')


# now to extract the data for each variable and region
# and combine into a single large numpy array for covariance calculation

data_values = {} # dict of dataframes  indexed by variable
loc_values = {} # dict of location values indexed by variable

for var,data  in timeseries.items():
    var_regn_names = [var+'_'+str(r) for r in data.region.values]
    dv = data.stack(dict(stack_dim=['time','realisation'])).compute().values.T
    data_values[var]=pd.DataFrame(dv,columns=var_regn_names)
    mn_data = mn_timeseries[var].mean('time').compute().values
    mn_data = pd.Series(mn_data,index=var_regn_names).rename(var)
    loc_values[var]=mn_data


# now to compute the covariance matrix -- applying the scaling to the data
cov_estimator = options['covariance_estimator'] # will raise an error if nothing provided. Defaults should have set something
fn = covariance_estimators[cov_estimator].fit
diagnostics=dict()
ordered_keys = sorted(data_values.keys()) # ordering we want.
if options.get('block',False): # block set and is True
    my_logger.info(f'Computing per variable covariance matrix with zero covariance between variables using {cov_estimator}')
    cov = pd.DataFrame()
    for var in ordered_keys:
        dv = data_values[var]
        # apply scaling!
        scaled_data = (dv*scale_units[var]).astype('double')
        if scaled_data.shape[1] < 2 and cov_estimator.startswith('GraphicalLasso'):
            fall_back = 'EmpiricalCovariance'
            my_logger.warning(f'Need > 2 features for {cov_estimator} for {indx} using {fall_back} instead')
            result = covariance_estimators[fall_back].fit(scaled_data)
            attr_value  = get_metadata(result,fall_back)
        else:
            result = fn(scaled_data)
            attr_value = get_metadata(result,cov_estimator)

        diagnostics.update({var:  (cov_estimator,attr_value)})
        cov_block = pd.DataFrame(result.covariance_, index=dv.columns, columns=dv.columns)
        cov = UKESMlib.merge_cov(cov_block, cov) # merge the covariance matrices

else:
    my_logger.info('Computing full covariance matrix allowing covariance between variables')
    cov_data = pd.concat([data_values[k]*scale_units[k] for k in ordered_keys],axis=1).astype('double')
    # concatenate all the data values.
    result =fn(cov_data)
    indx = cov_data.columns # ordering of the variables
    cov = pd.DataFrame(result.covariance_, index=indx, columns=indx)
    attr_value = get_metadata(result,cov_estimator) # might be None
    diagnostics.update({'All':attr_value})
# flatten the scale_units to a pandas series
scale_units = pd.concat([scale_units[k] for k in ordered_keys])
# now to rescale the covariance matrix back to SI units
scales_matrix = pd.DataFrame(np.outer(scale_units, scale_units),
                             index=scale_units.index, columns=scale_units.index) # used for reversing the scaling
cov /= scales_matrix # rescale back to SI units

indx = scale_units.index.tolist() # ordering of the variables
# reorder covariance matrix
cov = cov.reindex(index=indx, columns=indx)
# and the location values
loc_values = pd.concat([loc_values[k] for k in ordered_keys],axis=0).reindex(indx)


# deal with over rides!
if options.get('override_cov') is not None:
    for k,v in options['override_cov'].items():
        cov.loc[k,:] = 0.0
        cov.loc[:,k] = 0.0
        cov.loc[k,k] = v
        my_logger.debug(f'Set covariance {k} to {v} with all off diagonal values zero')

if options.get('override_loc') is not None:
    for k,v in options['override_loc'].items():
        loc_values.loc[k] = v
        my_logger.debug(f'Set location {k} to {v}')





# write out the location and covariance matrix
loc_values.to_json(options['location_file'], indent=2)
my_logger.info(f'Wrote location to {options["location_file"]}')
cov.to_csv(options['cov_file'], index_label='region')
my_logger.info(f"Wrote covariance matrix to {options["cov_file"]}")
# optional write out scaling factors
if options.get('scalings_file') is not None:
    with options['scalings_file'].open('wt') as fp:
        json.dump(scale_units[scale_units != 1.0].to_dict(), fp, indent=2)
    my_logger.info(f'Wrote scaling factors to {options["scalings_file"]}')


# print out the variables used if needed
if options.get('variables_file', False):
    with options['variables_file'].open('wt') as fp:
        json.dump(list(scale_units.index), fp, indent=2)


# deal with metadata file if needed. Note need to clean all optiosn so they are jsonable.
if options.get('metadata_file') is not None:
    metadata = dict(program = str(pathlib.Path(__file__).absolute()),
                    time_run = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d:%H-%M-%SZ'),
                    n_variables=len(timeseries),
                    variables={k:int(v.realisation.max()+1) for k,v in timeseries.items()},
                    n_files = len(files),
                    files = [str(f) for f in files],
                    diagnostics=diagnostics
                    )
    # fix options by converting pathlib Paths to strings
    options = {k:(str(v) if isinstance(v,pathlib.Path) else v) for k,v in options.items()}
    # and deal with files.
    options['files'] = [str(f) for f in files]
    metadata.update(options=options)
    if len(excluded_file_vars) >0:
        metadata.update(excluded_file_vars=excluded_file_vars)
        metadata.update(excluded_file_vars_comment="Dict indexed by variable of files excluded.")

    with pathlib.Path(options['metadata_file']).open('wt') as fp: # need to make it a path as converted everything to strings above.
        json.dump(metadata, fp, indent=2)
    my_logger.info(f'Wrote metadata to {options["metadata_file"]}')

