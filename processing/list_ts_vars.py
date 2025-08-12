#!/usr/bin/env python
# list all variables in the time series files.
import argparse
import xarray
import pathlib

parser = argparse.ArgumentParser(description="List all variables in the time series files.")
parser.add_argument("input_files", help="Input file containing time series data", type=pathlib.Path, nargs='+')
parser.add_argument('--attributes', action='store_true', help="List attributes of the variables")
args = parser.parse_args()

for input_file in args.input_files:
    if not input_file.exists():
        raise ValueError(f"Input file {input_file} does not exist.")

    ds = xarray.open_dataset(input_file)
    print(f"Variables in {input_file}:")
    for var_name in ds.data_vars:
        print(f"  - {var_name}")
    if args.attributes:
        print(f"Attributes in {input_file}:")
        for attr_name, attr_value in ds.attrs.items():
            print(f"  - {attr_name}: {attr_value}")

    print()  # Print a newline for better readability