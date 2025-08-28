#!/usr/bin/env python
"""
Process AATSR data to extract Aerosol Optical Depth (AOD) @ 550 nm and save to a NetCDF file.
"""
import xarray as xarray
import pandas as pd
import argparse
import pathlib
import UKESMlib
my_logger=UKESMlib.my_logger
def extract_aatsr_aod(file:pathlib.Path) -> xarray.DataArray:
    """
    Extract aatsr AOD data from a given netcdf file and return it as an xarray DataArray.

    Parameters:
    file (pathlib.Path): Path to the AATSR netcdf file. (Swansea University algorithm)

    Returns:
    xarray.DataArray: DataArray containing AOD data with time, latitude, and longitude coordinates.
    """
    my_logger.info(f"Extracting AATSR AOD data from {file}")
    ds = xarray.open_dataset(file, engine='netcdf4')
    var = 'AOD550_mean'
    aod = ds[var]

    # Extract time from the attributes
    time_str = ds.attrs['time_coverage_start']
    time = pd.to_datetime(time_str).tz_localize(None)  # Convert to datetime removing UTC.

    # Set coordinates
    aod = aod.assign_coords(time=time)
    aod = aod.rename('AOD_550').load()
    my_logger.debug(f'Extracted AOD data shape: {aod.shape}, time: {aod.time.values}')
    return aod

parser = argparse.ArgumentParser(description="Extract AATSR AOD data from netcdf files and save to a netCDF file.")
parser.add_argument('input_files', type=pathlib.Path, nargs='+',
                     help="Input AATSR netcdf files.")
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
# generate the files by expanding the inputs.
files=[]
for file in args.input_files:
    if file.is_file():
        files.append(file)
    else:
        files += sorted(file.parent.glob(file.name))
if not files:
    raise FileNotFoundError(f"No files found matching {args.input_files}")
my_logger.info(f"Processing {len(files)} files to extract AATSR AOD data")
# step 1 get the attributes from the first file
attrs = xarray.open_dataset(files[0], engine='netcdf4').attrs
# pop out the unneeded attributes
for attr in ['inputfilelist', 'time_coverage_start', 'time_coverage_end']:
    attrs.pop(attr, None)
aod=[extract_aatsr_aod(f) for f in sorted(files)]
aod = xarray.concat(aod, dim='time').sortby('time') # and sort by time
aod = aod.assign_attrs(attrs) # add back the attributes
aod.to_netcdf(args.output) # and save the file
my_logger.info(f'Saved AATSR AOD data to {args.output}')
