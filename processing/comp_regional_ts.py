#!/usr/bin/env python
"""
A  script for conservatively regridding and computing regional means from multiple NetCDF files.

Functions:
1. Conservatively regrid multiple NetCDF files on lon/lat coordinates to a land/ocean fraction grid.
2. Correctly handle static land files (without time dimension) and input data with time dimension.
3. Compute area-weighted averages for 10 regions.
4. Support processing specific variables.
5. Concatenate all data along the time dimension.
6. Add appropriate attributes and output as a NetCDF file.

"""

import argparse
import logging
import pathlib
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import xarray
import xarray_regrid # regridding in xarray
import glob

import UKESMlib


my_logger = logging.getLogger('comp_regional_ts')  # create a logger for this module


def conservative_regrid(source: xarray.Dataset, target: xarray.DataArray,
                        time_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None

                        ) -> xarray.Dataset:

    my_logger.info("Running conservative regridding...")
    regridded = dict()

    for var_name, var_data in source.data_vars.items():
        if not UKESMlib.is_lon_lat(var_data): # check is long/lat field
            my_logger.warning(f'{var_name} not long/lat field.. Skipping regridding for this variable.')
            continue # skip this variable as not long/lat.
        long_name,lat_name,vert_name,time_name = UKESMlib.guess_coordinate_names(var_data)
        rename={long_name:'longitude',lat_name:'latitude'} # rename the coords to be lon/lat/level/time
        if time_name is not None:  # if there is a time dimension
            rename.update({time_name: 'time'})  # rename time coord to 'time'
        da = var_data.rename(rename) # rename the coords to be lon/lat

        # now do the regird
        regridded[var_name] = da.regrid.conservative(target) # and regrid using xarray-regrid.

    regridded = xarray.Dataset(regridded)  # convert to a Dataset
    return regridded


def fix_units(da: xarray.DataArray) -> xarray.DataArray:
    """
    Fix units of the dataset. Converts the following units:
    - 'm' (per day) to 'kg/sec' for precipitation fields for ERA5 data
    - 'mm/day' to 'kg/sec'
    - 'mm/month' to 'kg/sec'
    - 'degrees Celsius' or 'degree C' to Kelvin ('K')
    - 'microns' to meters ('m')
    """
    attrs = da.attrs
    # Example: Convert units to a standard format if needed
    unit = da.attrs.get('units')
    lon,lat,vert,time = UKESMlib.guess_coordinate_names(da)
    # and standardise units
    da = da.rename({lon:'longitude',lat:'latitude'})
    if time is not None:  # if there is a time dimension
        da = da.rename({time: 'time'})
    if unit is None:
        return  da # nothing to do as have no units.
    with xarray.set_options(keep_attrs=True):
        if (unit == 'm') and attrs.get('GRIB_stepType') == 'avgad' and attrs.get('GRIB_name') == 'Total precipitation': # A precip field
            my_logger.info("Scaling ERA5 precip to kg/second from m/day")
            result = da * 1000 / (24 * 60 * 60.)  # convert to kg/sec
            result.attrs['units'] = 'kg/sec'  # update units to kg/sec
            my_logger.info("Scaling ERA5 precip to kg/second from m/day for {da.name}")
        elif unit == 'mm/day':
            result = da / (24 * 60 * 60)  # convert to kg/sec
            result.attrs['units'] = 'kg/sec'  # update units to kg/sec
            my_logger.info("Converting mm/day to kg/sec")
        elif unit == 'mm/month':
            result = da/(da['time'].dt.days_in_month * 24 * 60 * 60)  # convert to kg/sec
            result.attrs['units'] = 'kg/sec'  # update units to kg/sec
            result.attrs['units'] = 'kg/sec'  # update units to kg/sec
            my_logger.info("Converting mm/month to kg/sec for {da.name}")
        elif unit in ["degrees Celsius",'degree C']:  # convert to K
            result = da + 273.16
            result.attrs['units'] = 'K'  # update units to Kelvin
            my_logger.info(f"Converting degrees Celsius to Kelvin for {da.name}")
        elif unit in ['microns','um']:  # convert to meters
            result = da*1e-6
            result.attrs['units'] = 'm'  # update units to meters
            my_logger.info(f"Converting from microns to m for {da.name}")
        else:
               return da # nothing to do.
    return result






def process_files(input_files: list[str],
                  land_sea_file: pathlib.Path,
                  output_file: pathlib.Path,
                  critical_value: float = 0.5,
                  variables: Optional[list[str]] = None,
                  rename_vars: Optional[dict[str,str]] = None,
                  time_range:Optional[tuple[pd.Timestamp,pd.Timestamp]]=None) -> xarray.Dataset:
    my_logger.info("Starting file processing...")

    land_fract = xarray.load_dataarray(land_sea_file,decode_times=False).squeeze(drop=True).drop_vars(['surface','t'],errors='ignore')
    lon_name, lat_name, _,_ = UKESMlib.guess_coordinate_names(land_fract)
    land_fract = land_fract.rename({lon_name:'longitude',lat_name:'latitude'}).transpose('latitude','longitude') # in right order.

    # need to make this something that is specified
    masks = UKESMlib.create_region_masks(land_fract,  critical_value)
    if len(input_files) < 5:
        my_logger.info(f"Processing files: {input_files}")
    else:
        my_logger.info(f"Processing files: {' '.join(input_files[0:2])} ... {' '.join(input_files[-3:-1])}")
    with xarray.open_mfdataset(input_files) as ds:


        if variables:
            vars = set(variables) & set(ds.data_vars.keys())  # only keep those variables in the dataset
            if len(vars) == 0:
                raise ValueError(f"No variables found in the dataset matching {variables}.")
            ds = ds[vars]

        if rename_vars is not None:
            my_logger.info(f'Renaming variables: {rename_vars}')
            # only rename those variables in the data_vars
            names_got = set(ds.data_vars.keys())
            names_to_remove = set(rename_vars.keys()) - names_got
            for name in names_to_remove:
                rename_vars.pop(name)
            ds = ds.rename(rename_vars)



        if time_range is not None:
            for var_name, var_data in ds.data_vars.items():
                _,_,_,timec = UKESMlib.guess_coordinate_names(var_data)
                if timec is not None:
                    my_logger.info(f'Selecting time range: {time_range} for variable {var_name} using coord {timec}')
                    ds[var_name] = var_data.sel({timec:slice(*time_range)})


        my_logger.info(f'Processing {list(ds.data_vars.keys())} variables')
        
        #ds = ds.load() # force the load
        ds = ds.unify_chunks()
        with  np.errstate(divide='ignore', invalid='ignore'):
            regridded = conservative_regrid(ds, land_fract)  # regrid to land fraction grid

            regridded = regridded.load()
        regridded = regridded.map(fix_units)  # fix units of the regridded data
        logging.info('Regridded data')
        regional_avg = UKESMlib.compute_regional_averages(regridded,  masks)
        # add in non long/lat fields.
        regional_avg.update({ var_name:var_data.load() for var_name,var_data in ds.data_vars.items()
                              if not UKESMlib.is_lon_lat(var_data)})

        


    # copy attributes from the last ds we have
    bad_keys = ['_FillValue', 'missing_value','valid_min', 'valid_max','valid_range','actual_range','units'] # keys we do not want to copy over
    for var_name in regional_avg.data_vars: # variables
        attrs = ds[var_name].attrs.copy()
        for key in  bad_keys:
            attrs.pop(key,None)
        regional_avg[var_name].attrs.update(attrs) # update the attribues. Sigh meta-data

    attrs = ds.attrs.copy()
    for key in  bad_keys:
        attrs.pop(key,None)
    regional_avg.attrs.update(attrs) # update the attribues. Sigh meta-data

    regional_avg.attrs.update(attrs)
    regional_avg.attrs['description'] = 'Regional average data'
    regional_avg.attrs['script'] = __file__
    regional_avg.attrs['critical_value'] = str(critical_value)
    output_file.parent.mkdir(parents=True, exist_ok=True)  # ensure the output directory exists
    regional_avg.to_netcdf(output_file)

    my_logger.info("Processing complete.")
    return regional_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs='+',type=str)
    parser.add_argument('--variables', nargs='+',help='Variables to process',
                        default=[])
    parser.add_argument('--land_sea_file', required=True,type=pathlib.Path)
    parser.add_argument('--output_file', required=True,type=pathlib.Path)
    parser.add_argument('--critical_value', type=float, default=0.5,
                         help='Values >= this are land; less than Sea')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--overwrite', dest='overwrite', action='store_true', help='Enable overwrite')
    group.add_argument('--nooverwrite', dest='overwrite', action='store_false', help='Disable overwrite')
    parser.set_defaults(overwrite=False)
    parser.add_argument('--log_level', type=str, default='WARNING', help='Log level of of the script')
    parser.add_argument('--rename', type=str, nargs='+',default=None,
                        help='Pairs of variable names to rename in the form old_name:new_name, e.g. tas:temperature. Variable selection occurs before renaming.')
    parser.add_argument('--time_range', nargs=2, type=pd.Timestamp, default=None
                        , help='Time range to use for processing. If not provided, all times will be used.')
    args = parser.parse_args()
    my_logger = UKESMlib.setup_logging(rootname='comp_regional_ts',level = args.log_level)
    input_files = []
    for file in args.input_files:
        input_files += [f for f in glob.glob(file)]  # expand wildcards
    # check all input files exist
    missing_files = False
    for file in input_files:
        if not pathlib.Path(file).exists():
            my_logger.warning(f"Input file {file} does not exist.")
            missing_files = True
    if missing_files:
        raise ValueError("One or more input files do not exist. Please check the input files.")



    # Set up logging


    if (not args.overwrite) and args.output_file.exists():
        my_logger.warning(f'File {args.output_file} exists. See --overwrite to overwrite it')
        exit(0)
    rename_vars = args.rename
    if rename_vars is not None:
        rename_vars = dict([p for p in pair.split(':')] for pair in rename_vars)  # convert to dict

    if not args.land_sea_file.exists():
        raise ValueError(f"Land/Sea file {args.land_sea_file} does not exist.")
    process_files(input_files, args.land_sea_file, args.output_file,
                  critical_value=args.critical_value, variables=args.variables,rename_vars=rename_vars,
                  time_range =args.time_range)


if __name__ == '__main__':
    main()
