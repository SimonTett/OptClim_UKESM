#!/usr/bin/env python
"""
Extract UM pp data based on stash codes and time sampling.
example: extract_um_data.py /gws/nopw/j04/terrafirma/tetts/um_archive/u-dr157/19800701T0000Z/*a.p[m5]*.pp --stash m01s02i305 m01s30i206 m01s30i204 m01s34i114 m01s03i236 m01s34i073 m01s05i216 m01s34i102 m01s02i464 m01s02i330 m01s34i104 m01s02i304 m01s02i585 m01s02i453 m01s01i209 m01s16i222 m01s34i108 m01s02i205 m01s02i206 m01s01i208 m01s02i303 m01s02i451 m01s02i463 m01s02i452 m01s02i301 m01s34i072 m01s02i300 m01s05i063 --sampling 1 --output /gws/nopw/j04/terrafirma/tetts/um_archive/u-dr157/198007.nc --log_level DEBUG

"""
import argparse
import UKESMlib
import pathlib
import glob
import iris
import json
iris.FUTURE.save_split_attrs = True
parser = argparse.ArgumentParser(description='Extract pp data from UM files based on stash code and time sampling and write to netcdf file')
parser.add_argument("FILES",nargs='+',help='List of files to read')
parser.add_argument('--stash',help='List of stash codes to extract',nargs='+')
parser.add_argument('--samples',help='List of sampling times to extract',nargs='+',type=int)
parser.add_argument('--output',help='File to output to.')
parser.add_argument('--log_level',help='Set logging level',default='WARNING')
parser.add_argument('--overwrite',help='Overwrite existing file',type=bool)
parser.add_argument('--unlimited_dims',help='List of unlimited dimensions',nargs='+')
parser.add_argument('--select_file',
                    help='filepath to json file containing selection info. Will replace stash and sampling ',
                    type=pathlib.Path)

args = parser.parse_args()  # and parse the arguments

defaults=dict(
    log_level='WARNING',
    overwrite=False,
    unlimited_dims='time') # default values

config_args=dict(stash=args.stash,
                samples=args.samples,
                log_level=args.log_level,
                overwrite=args.overwrite,
                output=args.output,
                unlimited_dims=args.unlimited_dims
                )
config_args={k:v for k,v in config_args.items() if v is not None} # filter out the Nones
config=dict()
if args.select_file:
    with args.select_file.open('rt') as fp:
        config=json.load(fp)

config.update(config_args) # cmd args overrule json.
# and then defaults but only if value is None
config.update({k:v for k,v in defaults.items() if config.get(k) is None})

# setup logging
UKESMlib.setup_logging(args.log_level)
UKESMlib.my_logger.info(f'Reading data from {len(args.FILES)} files')
# log the config
for k,v in config.items():
    UKESMlib.my_logger.info(f'{k}:{v}')


if config.get('stash') is None and config.get('samples') is None:
    raise ValueError('No selection provided')

output=config.get('output')
if output is None:
    raise ValueError('Specify output')
else:
    output = pathlib.Path(output)

if output.suffix != '.nc':
    raise ValueError('Write to nc ')
if output.exists() and not config.get("overwrite",False):
    UKESMlib.my_logger.info(f'Exiting as output file {output} exists.')
    raise ValueError(f'{output} exists and overwrite is not set')
UKESMlib.my_logger.info(f'Reading data from {args.FILES}')
cubes = UKESMlib.um_cubes(args.FILES,stash_codes=config.get('stash'),intervals=config.get('samples')) # extract data
cubes.realise_data() # force the load as can get hdf error if write when not loaded... 
UKESMlib.my_logger.info(f'Writing data to {output}')
iris.save(cubes,output,unlimited_dimensions=config['unlimited_dims'],netcdf_format='NETCDF4',zlib=True) # write data






