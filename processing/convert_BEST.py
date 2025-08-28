#!/usr/bin/env python

"""
# Process BEST data to fix times and convert to absolute temperatures.
"""

import argparse
import logging
import pathlib
import xarray
import cftime
import sys
my_logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="""Process BEST data to fix netcdf issues and convert to absolute temperatures.
""")
parser.add_argument("input_file", help='Input file containing BEST data', type=pathlib.Path)
parser.add_argument("--output_file", help='Output file to write processed BEST data', type=pathlib.Path)
parser.add_argument("--log_level", type=str, default='warning', help='Log level of the script')
parser.add_argument('--overwrite', action='store_true', help='Enable overwrite of output file if it exists')

args = parser.parse_args()
logging.basicConfig(level=args.log_level.upper(), format='%(asctime)s - %(levelname)s - %(message)s')

best = xarray.open_dataset(args.input_file)
my_logger.info(f"Opened BEST data from {args.input_file}")
# work out output file and if it exists.
if args.output_file is None:
    output_file = args.input_file.parent/(args.input_file.stem + '_processed.nc')
else:
    output_file = args.output_file

if output_file.exists() and not args.overwrite:
    my_logger.warning(f"Output file {output_file} already exists. Use --overwrite to overwrite it.")
    sys.exit(0)
# Convert BEST data to absolute temperatures


with xarray.set_options(keep_attrs=True):
    time = time = [d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                   for d in  cftime.num2date((best.time - 1750.) * 365, 'days since 1750-01-01')]
    best = best.assign_coords(time=time)  # assign the time coordinate
    # and convert month_number to month.
    best['month_number'] = best['month_number']+1 # assign the month_number coordinate
    best = best.rename(month_number = 'month')  # rename month_number to month
    absolute_temp = best['temperature'].groupby('time.month') + best['climatology']
    best['absolute_temperature'] = absolute_temp.assign_attrs(
        long_name='Absolute Temperature',
        standard_name='air_temperature',
        valid_min=-100.0,
        valid_max=100.0
    ) # calling it tmn for compatability with CRU data.
    # extract to > 60S for compatability with CRU data
    best = best.sel(latitude=slice(-60,90))  # select latitudes > -60
## need to rename variables for compatibility with CRU data
# now to write the output file
history = best.attrs.get('history', '')
history += f' Processed BEST data to absolute temperatures using {pathlib.Path(__file__).name}.'
best.attrs['history'] = history


my_logger.info(f"Writing processed BEST data to {output_file}")
best.to_netcdf(output_file, format='NETCDF4')
my_logger.info("Processing complete.")
