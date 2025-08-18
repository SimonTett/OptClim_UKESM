import xarray
import xarray_regrid
import typing
import logging
import numpy as np
import socket
import pathlib
import os
import sys
import subprocess
import pandas as pd
import iris
import re
my_logger = logging.getLogger('UKESM')

## work out base-dirs for data depening on machine
host = socket.gethostname()
on_jasmin =re.match('sci-vm-0\d.jasmin.ac.uk',host)
try:
    base_dir = pathlib.Path(os.getenv('BASE_DIR'))
except TypeError as e: # failed coz BASE_DIR does not exist
    my_logger.warning('BASE_DIR not in env')
    if host.startswith('GEOS-W'):  # Geos windows desktop
        base_dir = pathlib.Path(r"P:\optclim_data")
    elif on_jasmin:
        base_dir=pathlib.Path('gws/nopw/j04/terrafirma/tetts/data')

    else:
        raise ValueError('Do not know how to define base_dir.  Define BASE_DIR or modify code')

try:
    process_dir = pathlib.Path(os.getenv('PROCESS_DIR'))
except TypeError as e:  # failed coz PROCESS_DIR does not exist
    my_logger.warning('PROCESS_DIR not in env')
    if host.startswith('GEOS-W') or host.startswith('GEOS_L'):  # Geos windows dekstop or laptop
        process_dir = pathlib.Path(r"C:\Users\stett2\OneDrive - University of Edinburgh\data\Opt_UKESM1")
    elif on_jasmin:
        process_dir = base_dir/'processing'
    else:
        raise ValueError('Do not know how to define process_dir.  Define PROCESS_DIR or modify code')


def setup_logging(level: typing.Optional[typing.Union[int, str]] = None,
                  rootname: typing.Optional[str] = None,
                  log_config: typing.Optional[dict] = None):
    """
    Setup logging.
    :param: level: level of logging. If None logging.WARNING will be used
    :param: rootname: rootname for logging. if None OPTCLIM will be used.
    :param: log_config config dict for logging.config --
          see https://docs.python.org/3/library/logging.config.html
          If not None will only be used if level is not None and the actual
          value of level will be ignored.
    """

    if rootname is None:
        rootname = 'UKESM'

    logger = logging.getLogger(rootname)  # get  root logger

    # need both debugging turned on and a logging config
    # to use the logging_cong
    if level is not None and log_config is not None:
        logging.debug("Using log_config to set up logging")
        logging.config.dictConfig(log_config)  # assume this is sensible
        return logger

    if level is None:
        level = logging.WARNING

    # set up a sensible default logging behaviour.

    logger.handlers.clear()  # clear any existing handles there are
    logger.setLevel(level)  # set the level

    console_handler = logging.StreamHandler()
    fmt = '%(asctime)s %(levelname)s:%(name)s:%(funcName)s: %(message)s'
    formatter = logging.Formatter(fmt)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)  # turning this on gives duplicate messages.
    logger.propagate = False  # stop propogation to root level which suppresses duplicate messages.
    # see https://jdhao.github.io/2020/06/20/python_duplicate_logging_messages/
    return logger

def init_log(
        log: logging.Logger,
        level: str,
        log_file: typing.Optional[typing.Union[pathlib.Path, str]] = None,
        datefmt: typing.Optional[str] = '%Y-%m-%d %H:%M:%S',
        mode: str = 'a'
) -> logging.Logger:
    """
    Set up logging on a logger! Will clear any existing logging.
    :param log: logger to be changed
    :param level: level to be set.
    :param log_file:  if provided pathlib.Path to log to file
    :param mode: mode to open log file with (a  -- append or w -- write)
    :param datefmt: date format for log.
    :return: nothing -- existing log is modified.
    """
    log.handlers.clear()
    log.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s:  %(message)s',
                                  datefmt=datefmt
                                  )
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    # add a file handler.
    if log_file:
        if isinstance(log_file, str):
            log_file = pathlib.Path(log_file)
        log_file.parent.mkdir(exist_ok=True, parents=True)
        fh = logging.FileHandler(log_file, mode=mode + 't')  #
        fh.setLevel(level)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    log.propagate = False
    return log

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
        cmd2 = [] # convert to strings
        for c in cmd:
            c2 = str(c)
            if (' ' in c2) or ('-' in c2 and not c2.startswith('-')):  # if there is a space or - in the command
                c2 = '"'+c2+'"'  # put it in quotes

            cmd2.append(c2)
        my_logger.warning(f"Command {' '.join(cmd2)} failed with return code {result.returncode}")
        breakpoint()


    return result

def guess_coordinate_names(da: xarray.DataArray) -> \
        tuple[typing.Optional[str],typing.Optional[str],typing.Optional[str],typing.Optional[str]]:
    possible_lon_names = ['longitude','longitude_0','longitude_1', 'lon', 'x', 'long']
    possible_lat_names = ['latitude', 'latitude_0','latitude_1','lat', 'y']  # TODO -- rewrite to use regexps
    possible_time_names = ['time', 't', 'date', 'valid_time']
    possible_vert_names = ['atmosphere_hybrid_sigma_pressure_coordinate',
                           'altitude', 'pressure', 'air_pressure', 'depth','level','z','Z',
                     'model_level_number','pressure_level']
    
    lon_name = next((n for n in possible_lon_names if n in da.dims), None)
    lat_name = next((n for n in possible_lat_names if n in da.dims), None)
    vert_name = next((n for n in possible_vert_names if n in da.dims), None)
    time_name = next((n for n in possible_time_names if n in da.dims), None)
    
    # if not lon_name or not lat_name:
    #    raise ValueError("Cannot automatically determine longitude/latitude coordinate names.")
    
    return lon_name, lat_name, vert_name,time_name


def is_lon_lat(da: xarray.DataArray):
    lon, lat, _,_ = guess_coordinate_names(da)
    ok = ((lon in da.dims) and (lat in da.dims))
    return ok


def conservative_regrid(source: typing.Union[xarray.Dataset,xarray.DataArray],
                        target: xarray.DataArray
                        ) -> xarray.Dataset:
    # TODO - make this only work for data-arrays and then use map for datasets.
    # Will return None if field is not long/lat.
    # And if field long.lat grid matches target -- just return it. No need for any more processing at that point.
    my_logger.debug(f"Regridding {source.name}")
    # 1 -- check all are long/lat fields.
    if isinstance(source,xarray.Dataset):
        for var_name, var_data in source.data_vars.items():
            if not is_lon_lat(var_data):
                raise ValueError(f'{var_name} not long/lat field.. ')
    elif isinstance(source,xarray.DataArray):
         if not is_lon_lat(source):
            raise ValueError(f'{source.name} not long/lat field.. ')
    else:
        pass
             



    regridded = source.regrid.conservative(target)  # and regrid using xarray-regrid.

    return regridded


def create_region_masks(land_fract: xarray.DataArray,
                        critical_value: float = 0.5) -> dict[str, xarray.DataArray]:
    my_logger.info("Creating regional masks...")
    lon_name, lat_name, _,_ = guess_coordinate_names(land_fract)
    tropics_boundary = 30.0
    latitude = land_fract[lat_name]
    tropics_mask = (np.abs(latitude) <= tropics_boundary)
    nh_extratropics_mask = latitude > tropics_boundary
    sh_extratropics_mask = latitude < -tropics_boundary

    land_mask = land_fract >= critical_value
    sea_mask = land_fract < critical_value

    masks = {
        'NHX_L': nh_extratropics_mask & land_mask,
        'NHX_S': nh_extratropics_mask & sea_mask,
        'T_L': tropics_mask & land_mask,
        'T_S': tropics_mask & sea_mask,
        'SHX_L': sh_extratropics_mask & land_mask,
        'SHX_S': sh_extratropics_mask & sea_mask,
        'NHX': nh_extratropics_mask,
        'T': tropics_mask,
        'SHX': sh_extratropics_mask,
        'global': xarray.ones_like(land_fract, dtype=bool)
    }
    return masks


def compute_area_weights(da: xarray.DataArray | xarray.Dataset) -> xarray.DataArray:
    """
    Use cos(lat) weights instead of true area weights (suitable for regular grids).
    """

    lon_name, lat_name, _,_ = guess_coordinate_names(da)
    lat = da[lat_name]

    weights = np.cos(np.deg2rad(lat))
    return weights


def da_regional_avg(da:xarray.DataArray,masks: dict[str, xarray.DataArray]) -> xarray.DataArray:
    my_logger.debug(f"Computing regional averages for {da.name}")
    result = []
    area_weights = compute_area_weights(da)
    lon_name, lat_name, _,_ = guess_coordinate_names(da)
    for region_name, mask in masks.items():
        masked_var = da.where(mask)
        mn = masked_var.weighted(area_weights.where(mask,0.0)).mean(dim=[lon_name, lat_name], skipna=True).squeeze(drop=True)
        mn = mn.expand_dims(region=[region_name])
        mn = mn.load()  # force dask to compute this.
        result.append(mn)
    result = xarray.concat(result, dim='region',coords='minimal')
    return result

def compute_regional_averages(ds: xarray.Dataset,
                              masks: dict[str, xarray.DataArray],
                              ) -> xarray.Dataset:
    my_logger.info("Computing regional averages...")
    results = {}

    for var_name, var_data in ds.data_vars.items():
        if not is_lon_lat(var_data):
            logging.debug(f'{var_name} is not a long/lat field. Skipping')
            continue



        results[var_name] = da_regional_avg(var_data,masks)

    return xarray.Dataset(results)

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
    mean_seas = mean_seas.dropna('time')##.drop_dims('nbounds', errors='ignore') # for datataray can't drop dims
    return mean_seas

def mslp_process(ts: xarray.DataArray) -> xarray.DataArray:
    """
    Special processing for specific datasets.
    Handles SLP for now.
    :param ts: xarray Dataset to process.
    :return: processed xarray Dataset.
    """

    if str(ts.name).lower() not in  ['slp','mslp','mean_sea_level_pressure']:
        my_logger.warning(f'Variable {ts.name} is not a recognized SLP variable. Processing as is.')
    my_logger.info(f'Processing SLP variable {ts.name} to convert to difference from global mean.')
    # remove the global mean and add_delta to the region names.
    result = (ts-ts.sel(region='global').values).where(ts.region != 'global', drop=True)
    region_names = [r+'_DGM' for r in result.coords['region'].values ]
    result = result.assign_coords(region=region_names)  # rename the regions

    return result
def process(ts:xarray.Dataset,
            mslp_vars:typing.Optional[list[str]]=None
            ) -> xarray.Dataset:
    # do data processing on the time series.

    if mslp_vars is not None:
        my_logger.info('Special processing enabled. ')
        ts = ts.merge(ts[mslp_vars].map(mslp_process))  # apply special processing to each variable


    resamp = ts.resample(time='YS-JAN')
    mean = resamp.mean().where(resamp.count() == 12)  # compute annual means where there are 12 months of data
    #mean = mean.dropna('time').drop_dims('nbounds', errors='ignore')
    mean = mean.dropna('time')
    # add in  seasonal cycle.
    ts_delta = seasonal_cycle(ts, 'jja') - seasonal_cycle(ts, 'djf')
    rgn_names = {r:r+'_'+'seas' for r in ts_delta.coords['region'].values}
    ts_delta= ts_delta.assign_coords(region = [rgn_names.get(r,r) for r in ts.coords['region'].values])
    result = xarray.concat([mean, ts_delta], dim='region').load()
    return result

def merge_cov(covariance: pd.DataFrame, covariance2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two covariance matrices, filling in missing values with zeros.
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


def um_cubes(files:typing.Union[list[str],str],
                 stash_codes:typing.Optional[list[str]]=None,
                 intervals:typing.Optional[list[int]]=None,
             inspect:bool = False,
                 ):
    """
    Extract cubes from UM file(s) based on  stash cpdes and (optionally) intervals
    It is a lot faster than std load which looks to work cubes for everything in the file. This can be very very slow... 
    files: list of files to process. Each one should be readable by iris.fileformats.pp.load
    stash_codes: list of strings of stash codes to extract. As long as your lsit is small then this function will save you time
    intervals: Sampling intervals to filter on -- if provided. 
    """
    def count_fields(fields):
        counts=dict()
        for f in fields:
            key=f'stash:{f.stash} lbtim:{f.lbtim} ia:{f.lbtim.ia} ib:{f.lbtim.ib} '
            count = counts.get(key,0)+1
            counts[key]=count
        return counts
    from  iris.fileformats.pp import load,load_pairs_from_fields
    if isinstance(files,str):
        files=[files]
    cubes_f = []
    for file in files:
        my_logger.debug(f'Reading from {file}')
        fields = list(load(file))
        if inspect: # print everything out
            counts = count_fields(fields)
            print(f"stash codes & lbtim in {file}")
            for k in sorted(counts.keys()):
                print(f'{k} has {counts[k]} fields')
            return None
        if stash_codes is not None: # got some stash codes
            fields = [f for f in fields if  str(f.stash) in stash_codes] # filter on stash codes.

        if intervals is not None: # got some intervals
             fields = [f for f in fields if (f.lbtim.ib !=2  or (f.lbtim.ib == 2 and  f.lbtim.ia in intervals))]
        my_logger.debug(f'Read {len(fields)} fields after filtering')
        new_cubes = [c for c,f in load_pairs_from_fields(fields) ] 
        my_logger.debug(f'Converted to {len(new_cubes)} cubes')
        cubes_f += new_cubes

    my_logger.debug(f'Have {len(cubes_f)} cubes')
    cubes = iris.util.combine_cubes(cubes_f)
    my_logger.debug(f'Combined to {len(cubes)} cubes')
    return cubes
