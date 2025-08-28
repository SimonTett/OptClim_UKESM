#!/usr/bin/env python
"""
Extract MODIS AOD data from hdf files and save to a netCDF file.
Longitude/latitude are set and time assigned based on the file name
"""

import UKESMlib
import xarray
import pathlib
import pandas as pd
import argparse
my_logger=UKESMlib.my_logger
def extract_modis_aod(file:pathlib.Path) -> xarray.DataArray:
    """
    Extract MODIS AOD data from a given HDF file and return it as an xarray DataArray.

    Parameters:
    file (pathlib.Path): Path to the MODIS HDF file.

    Returns:
    xarray.DataArray: DataArray containing AOD data with time, latitude, and longitude coordinates.
    """
    my_logger.info(f"Extracting MODIS AOD data from {file}")
    ds = xarray.open_dataset(file, engine='netcdf4')
    var = 'AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean'
    aod = ds[var]

    # Extract time from the filename
    time_str = file.stem.split('.')[1]  # Assuming the time is in the 2nd cpt of the filename
    time = pd.to_datetime(time_str, format='A%Y%j')  # Convert to datetime

    # Set coordinates
    aod = aod.assign_coords(time=time).rename({'YDim:mod08':'latitude', 'XDim:mod08':'longitude'})

    aod = aod.assign_coords(latitude=ds['YDim'].values)
    aod = aod.assign_coords(longitude=ds['XDim'].values)
    aod = aod.rename('AOD_550').load()
    my_logger.debug(f'Extracted AOD data shape: {aod.shape}, time: {aod.time.values}')
    return aod

parser = argparse.ArgumentParser(description="Extract MODIS AOD data from HDF files and save to a netCDF file.")
parser.add_argument('input_files', type=pathlib.Path, nargs='+',
                     help="Input MODIS HDF files.")
parser.add_argument('--output', type=pathlib.Path,
                    required=True, help="Output netCDF file to save the extracted AOD data.")
parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--log_level', type=str, default='INFO',
                    choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
                    help="Set the logging level ")
args = parser.parse_args()
UKESMlib.init_log(my_logger, level=args.log_level)
if (not args.overwrite) and args.output.exists():
    my_logger.warning(f"Output file {args.output} already exists. Use --overwrite to overwrite it.")
    exit(0)

aod=[extract_modis_aod(f) for f in sorted(args.input_files)]
aod = xarray.concat(aod, dim='time')
aod.to_netcdf(args.output) # and save the file
my_logger.info(f'Wrote data to {args.outut}')
