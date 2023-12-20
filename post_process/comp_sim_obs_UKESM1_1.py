#!/usr/bin/env python

""" Compute simulated observables using xarray.
xarray appears to be a lot faster than iris!
 Based on comp_obs. Hopefully faster 
Observations generated are in genProcess (or help for script)

 """

import argparse  # parse arguments
import json  # get JSON library
import os  # OS support
import pathlib
import re
import time
import typing
import numpy as np
import xarray
import iris
import logging

import StudyConfig

_name_pat = None


def new_name(name: str) -> str:
    """
    Generate a new name by incrementing the digits by 1 where name is of form text_N.
       If name is of form text return text_1
    :param name: Name to be incremented.
    :return:  Incremented name.
    """
    global _name_pat
    if _name_pat is None:
        _name_pat = re.compile(
            r'^(.*?)_(\d+)$')  # should match _digits at the end with the digits as one of the matches
    match = re.search(_name_pat, name)
    if match is not None:  # increase number
        digit = int(match.group(2))
        base = match.group(1)
    else:
        digit = 0
        base = name

    digit += 1  # increase the count by 1
    return base + f'_{digit}'  # return name + count.


def change_name(var: typing.Union[xarray.DataArray, xarray.Variable], clear: bool = False) -> str:
    """

    :param var: Xarray Variable or DataArray to see if already have
    :return: name or a new name. Any " " will be converted to _
    """
    global _named_vars
    if clear:
        _named_vars = dict()  # reset the global var
    name: str = var.name.replace(" ", "_")
    # convert  all spaces to "_" to make subsequent work easier

    while True:
        value = _named_vars.get(name)
        if value is None:  # not got this value.
            _named_vars[name] = var  # store the variable
            return name  # just return the name
        elif var.equals(value):  # we are the same so safe to reuse. Return name
            return name
        else:  # generate a new name and continue looping
            name = new_name(name)


def read_UMfiles(files: typing.Iterable[pathlib.Path]) -> xarray.Dataset:
    """
    Raad a bunch of UM  files and convert them to  a dataset
    :param files: iterable of files to read
    :return: dataset
    """
    data_array_list = []
    cubes = iris.load(files)
    for cube in cubes:
        da = xarray.DataArray.from_iris(cube).rename(cube.name())
        try:
            da = da.expand_dims(dim='time')  # attemp to expand time. Will fail if time already exists
        except ValueError:
            pass
        data_array_list.append(da)
    dataSet = merge_dataArray(data_array_list)
    return dataSet


def merge_dataArray(dataArray_list: typing.List[xarray.DataArray]) -> xarray.Dataset:
    """
    Merge (renaming list of dataArrays. dataArarrays and co-ords will be renamed if they have duplicated names but different values  )
    :param dataArray_list: List of dataArrays to be merged
    :return: Dataset of merged data
    """
    cleaned = []
    clear = True  # always clear first time
    for indx, da in enumerate(dataArray_list):  # loop over list
        rename_dict = dict()  # set up dict for renaming
        name = change_name(da, clear=clear)  # get the potentially new name for the dataarray
        clear = False  # do not clear next time around
        if name != da.name:  # change of name needed?
            da = da.rename(name)  # renaming the dataArray
        for cname in da.coords:  # loop over coords
            var = da[cname]
            var = var.drop_vars(var.coords)  # drop all coords on the co-ord.
            name = change_name(var)  # get potential new name
            if name != cname:  # need to change the name? Update the co-ords
                rename_dict.update({cname: name})
        if rename_dict: # got sommething to rename?
            logging.info(f"renaming  using {rename_dict}")  # log!
            da = da.rename(rename_dict)
        cleaned.append(da)  # rename

    dataset = xarray.merge(cleaned, compat='identical')
    dataset.attrs = dict(history='merged dataset')
    return dataset


def expand(filestr: str) -> pathlib.Path:
    """

    Expand any env vars, convert to path and then expand any user constructs.
    :param filestr: path like string
    :return:expanded path
    """
    path = os.path.expandvars(filestr)
    path = pathlib.Path(path).expanduser()
    return path


def guess_lat_lon_vert_names(dataArray):
    """
    Guess the names of latitude, longitude & vertical co-ords in the dataArray
    starts with latitude/longitude, then lat/lon, then latitude_N/longitude_N then lat_N,lon_N
    and with atmosphere_hybrid_sigma_pressure_coordinate, altitude, air_pressure for vertical

    :param dataArray:
    :return: latitude, longitude & vertical co-ord names
    """

    def find_name(dimensions, patterns):
        name = None
        for pattern in patterns:
            reg = re.compile("^" + pattern + r"(_\d+)?" + "$")
            for d in dimensions:
                if reg.match(d):
                    name = d
                    break
        return name

    dims = dataArray.dims
    lat_patterns = ['latitude', 'lat']
    lon_patterns = ['longitude', 'lon']
    vert_patterns = ['atmosphere_hybrid_sigma_pressure_coordinate', 'altitude', 'pressure', 'air_pressure',
                     'model_level_number']
    lat_name = find_name(dims, lat_patterns)
    lon_name = find_name(dims, lon_patterns)
    vert_name = find_name(dims, vert_patterns)

    return lat_name, lon_name, vert_name


def model_delta_p(pstar, a_bounds, b_bounds):
    """
    Compute the delta_pressures for each model level for UM hybrid-sigma grid
    For which pressure is a+b*pstar.
    :param pstar: Surface pressure
    :param a_bounds: bounds of "a" co-efficients
    :param b_bounds: bounds of "b" co-efficients
    :return: Pressure thicknesses as a function of vertical co-ord for each model grid cell
    """

    bnds_coord = [d for d in a_bounds.dims if d.startswith('bounds')][0]
    delta = lambda bnds: bnds.sel({bnds_coord: 1}) - bnds.sel({bnds_coord: 0})
    delta_a = delta(a_bounds)
    delta_b = delta(b_bounds)
    delta_p = delta_a + pstar * delta_b

    return delta_p


def total_column(data: typing.Optional[xarray.DataArray],
                 atmos_mass: typing.Optional[xarray.DataArray],
                 scale: typing.Optional[float] = None):
    """
    Compute total column of substance (kg/m^2) assuming hydrostatic atmosphere.
        (Atmospheric mass in layer is \Delta P/g)
    :param data: dataArray of the variable for which column calculation is being done. (or None)
        Assumed to be a mass mixing ratio (kg/kg)
    :param atmos_mass: total mass of atmosphere.
    :param scale:  scale factor.If None no scaling is done
    :return: total_column of substance
    """
    if data is None or atmos_mass is None:
        return None
    lat, lon, vertical_coord = guess_lat_lon_vert_names(data)
    area = np.cos(np.deg2rad(data[lat])) * 6371e3  # simple cos lat weighting.
    col = (data * atmos_mass).sum(vertical_coord) / area  # work out column  by integrating and dividing by area.
    if scale is not None:
        col *= scale

    return col


def names(dataset, name=None):
    """
     Return dictionary of standard (or long or stash)  names for each variable in a dataset.

    :param dataset: xarray data
    :param name: what you want to return (None, 'standard','long'). If None (default) then standard_name will be returned
    :return: dict of standard names
    """
    allowed_names = ['standard', 'long', 'stash']
    if name is not None and name not in allowed_names:
        raise ValueError(f"Unknown name {name}")
    if name is None or name == 'standard':
        key = 'standard_name'
    elif name == 'long':
        key = 'long_name'
    elif name == 'stash':
        key = 'STASH'
    else:
        raise Exception(f"Do not know what to do with {name}")

    lookup = {}
    for var in dataset.variables:
        try:
            name = dataset[var].attrs[key]
            if key == 'STASH':  # convert stash to print repr.
                name = f"m{name.model:02d}s{name.section:02d}i{name.item:03d}"
            lookup[name] = var  # if not present then won't update lookup
        except KeyError:
            pass

    return lookup


def reff(dataset):
    reffwt = dataset.get('m01s02i463')  # COSP MODIS weighted liquid Reff
    wt = dataset.get('m01s02i452')  # COSP MODIS weight
    if (reffwt is None) or (wt is None):
        logging.warning("Failed to find reffwt or wt")
        return None
    return reffwt / wt


def genProcess(dataset:xarray.Dataset, land_mask:xarray.Dataset) -> typing.Dict[str,xarray.DataArray]:
    """
    Setup the processing information
    :param dataset: the dataset containing the data
    :param land_mask -- land mask as a dateArray.
    :return: dict containing data to be processed.
    """
    # create long & standard name lookup
    lookup_std = names(dataset)
    lookup_long = names(dataset, name='long')
    lookup_stash = names(dataset, name='stash')

    def name_fn(name: typing.Union[str, list[str]],
                dataset: xarray.Dataset,
                *args,
                name_type: typing.Optional[str] = None,
                **kwargs) -> (xarray.DataArray, None):
        """
        Lookup name/variable and then return datarray corresponding to it.
        If list provided iterate over. Any None return None; and then sum values.
        If not present return None
        :param long_name:
        :param **kwargs: Remaining kw args passed to select
        :param dataset:
        :return:
        """
        # handle list
        if isinstance(name, list):  # loop over vars calling name_fn and then add
            results = []
            for n in name:
                var = name_fn(n, dataset, *args, name_type=name_type, **kwargs)
                if var is None:
                    return None
                results.append(var)

            result = results[0].squeeze().copy()
            for r in results[1:]:
                result += r.squeeze()
            return result  # computed result

        if name_type is None or name_type == 'name':
            if name in dataset.variables:
                var = name
            else:
                var = None
                name_type = 'name'
        elif name_type == 'long':
            var = lookup_long.get(name)
        elif name_type == 'standard':
            var = lookup_std.get(name)
        elif name_type == 'stash':
            var = lookup_stash.get(name)

        else:
            raise ValueError(f"Do not know what to do with name_type {name_type}")

        if var is None:  # failed to find name so return None
            logging.warning(f"Failed to find name {name} of type {name_type}")
            return None
        da = dataset[var]
        if (len(args) > 0) or (len(kwargs) > 0):
            try:
                da = da.sel(*args, **kwargs)
            except KeyError:  # failed to find some co-ords
                logging.warning(f"Failed to select {var} using {args} or {kwargs}")
                return None
        return da

    latitude_coord, lon, vert = guess_lat_lon_vert_names(dataset[lookup_stash['m01s03i236']])  # 1.5 m air tempo.
    constrain_60S = dataset[latitude_coord] >= -60.
    # need to extract the actual values...
    constrain_60S = constrain_60S.latitude[constrain_60S]
    # set up the data to be meaned. Because of xarray's use of dask no processed  happens till
    # spatial (And temporal ) means computed. See means.
    mass = name_fn('m01s50i063', dataset, name_type='stash')
    process = {
        'OLR': name_fn('toa_outgoing_longwave_flux', dataset, name_type='standard'),
        'OLRC': name_fn('toa_outgoing_longwave_flux_assuming_clear_sky', dataset, name_type='standard'),
        'RSR': name_fn('toa_outgoing_shortwave_flux', dataset, name_type='standard'),
        'RSRC': name_fn('toa_outgoing_shortwave_flux_assuming_clear_sky', dataset, name_type='standard'),
        'INSW': name_fn('toa_incoming_shortwave_flux', dataset, name_type='standard'),
        'LAT': xarray.where(land_mask, dataset[lookup_stash['m01s03i236']], np.nan).sel(
            {latitude_coord: constrain_60S}),
        'Lprecip': xarray.where(land_mask, dataset[lookup_std['precipitation_flux']], np.nan).sel(
            {latitude_coord: constrain_60S}),
        'MSLP': dataset[lookup_std['air_pressure_at_sea_level']],
        'Reff': reff(dataset),
        # SO2 related. Except SO2_col the rest are for HadXM3 and will need updating.
        'SO2_col': total_column(name_fn('mass_fraction_of_sulfur_dioxide_in_air', dataset, name_type='name'),
                                mass),
        'dis_col': total_column(name_fn("SO4 DISSOLVED AEROSOL AFTER TSTEP", dataset, name_type='long'), mass),
        'aitkin_col': total_column(name_fn('SO4 AITKEN MODE AEROSOL AFTER TSTEP', dataset, name_type='long'), mass),
        'accum_col': total_column(name_fn('SO4 ACCUM. MODE AEROSOL AFTER TSTEP', dataset, name_type='long'), mass),
        'DMS_col': total_column(name_fn('mass_fraction_of_dimethyl_sulfide_in_air', dataset, name_type='name'),
                                mass),
        'Trop_SW_up': name_fn('tropopause_upwelling_shortwave_flux', dataset),
        'Trop_SW_net': name_fn('tropopause_net_downward_shortwave_flux', dataset),
        'Trop_LW_up': name_fn('tropopause_upwelling_longwave_flux', dataset),
        'Trop_LW_net': name_fn('tropopause_net_downward_longwave_flux', dataset),

    }
    # deal with T and rh values
    coord_500hPa = dict(pressure=500)  # co-ord value for 500 hPa
    coord_50hPa = dict(pressure=50)  # co-ord value for 50 hPa

    p_wt = name_fn('Heavyside function on pressure levels', dataset, name_type='long')
    if p_wt is not None: # got the p_wts
        temp = name_fn('m01s30i204', dataset, name_type='stash')
        if temp is not None:
            temp /= p_wt # scale by time above sfc.
            process.update(
                {'TEMP@50': temp.sel(coord_50hPa),
                 'TEMP@500': temp.sel(coord_500hPa)})

        # compute RH
        rh = name_fn('m01s30i206', dataset, name_type='stash')
        if rh is not None:
            rh /= p_wt # scale by time above sfc
            process.update(
                {'RH@500': rh.sel(coord_500hPa),
                 'RH@50': rh.sel(coord_50hPa), })
    # deal with SO2


    process['netflux'] = process['INSW'] - process['RSR'] - process['OLR']

    return process


def means(dataArray, name):
    """ 
    Compute means for NH extra tropics, Tropics and SH extra Tropics. 
    Tropics is 30N to 30S. NH extra tropics 30N to 90N and SH extra tropics 90S to 30S 
    Arguments:
        :param dataArray -- dataArray to be processed
        :param name -- name to call mean.



    """

    latitude_coord, lon, vert = guess_lat_lon_vert_names(dataArray)

    wt = np.cos(np.deg2rad(dataArray[latitude_coord]))  # simple cos lat weighting.
    # constraints and names for regions.
    constraints = {
        'GLOBAL': None,
        'NHX': lambda y: y > 30.0,
        'TROPICS': lambda y: (y >= -30) & (y <= 30.0),
        'SHX': lambda y: y < -30.0,
    }
    means = dict()
    for rgn_name, rgn_fn in constraints.items():
        if rgn_fn is None:
            v = dataArray.squeeze().weighted(wt).mean()
        else:
            msk = rgn_fn(dataArray[latitude_coord])  # T where want data
            v = dataArray.where(msk, np.nan).squeeze().weighted(wt.where(msk, 0.0)).mean()
        means[name + '_' + rgn_name] = float(v.load().squeeze().values)
        # store the actual values -- losing all meta-data

    return means  # means are what we want


def compute_values(files: typing.Iterable[pathlib.Path],
                   output_file: pathlib.Path,
                   start_time: typing.Optional[str] = None,
                   end_time: typing.Optional[str] = None,
                   land_mask_fraction:float = 0.5) -> typing.Dict:
    """

    :param files: iterable of files to readin
    :param output_file: output file to write to
    :param start_time: start time as iso date/time
    :param end_time:  end_time as iso date/time
    :return: dict of results
    """
    dataset = read_UMfiles(files)
    dataset = dataset.sel(time=slice(start_time, end_time))
    land_mask = dataset.land_area_fraction.isel(time=0).squeeze()  # land/sea mask
    land_mask = xarray.where(land_mask > land_mask_fraction, True, False)
    process = genProcess(dataset, land_mask)

    # now to process all the data making output.
    results = dict()
    for name, dataArray in process.items():
        if dataArray is None:  # no dataarray for this name
            logging.warning(f"{name} is None. Not processing")
            continue
        mean = means(dataArray, name)  # compute the means
        results.update(mean)  # and stuff them into the results dict.
        logging.debug(f"Processed {name} and got {mean}")

    # now fix the MSLP values. Need to remove the global mean from values and the drop the SHX value.
    results.pop('MSLP_SHX')
    for k in ['MSLP_NHX', 'MSLP_TROPICS']:
        results[k + '_DGM'] = results.pop(k) - results['MSLP_GLOBAL']

    # now to write the data
    with open(output_file, 'w') as fp:
        json.dump(results, fp, indent=2)


    return results


def do_work():
    """
    parse input arguments and run code. run cmd with  -h option to see what cmd line arguments are.
    uses the config file postProcess block with the following options:

        dir -- Path to where data is relative to cwd.
           Only used if cmd line arg dir not specified with default value  'share/data/History_Data/'
        file_pattern -- glob pattern for files to read. Default is 'a.py*.pp' which will process annual mean pp.
        start_time -- start time as iso-format string for data selection. Default None
        end_time -- end time as iso-format string for data selection. Default None
        outputPath -- if OUTPUT not provided on cmd line will define output file. Must be provided if OUTPUT not provided.
        land_mask_fraction -- fraction of area above which region is land. Default is 0.5


    """

    parser = argparse.ArgumentParser(description="""
    Post process Unified Model data to provide 32 simulated observations. Example use is:
    
    comp_sim_obs_UKESM1_1.py input.json /exports/work/geos_cesd_workspace/OptClim/Runs/st17_reg/01/s0101/A/output.json *a.py*.pp
    
    Observations are:
    Global mean, Northern Hemisphere Extra-tropical average (30N-90N), Southern Hemisphere Extra-tropical (90S-30S) mean and 
    Tropical (30S - 30N) mean for:
      Temperature at 500 hPa,
      Relative Humidity at 500 hPa
      Outgoing Longwave Radiation
      Outgoing Clearsky Longwave Radiation
      Reflected Shortwave Radiation
      Clear sky reflected shortwave
      Land air temperature at 1.5m (north of 60S)
      Land precipitation at 1.5m  (north of 60S)
      Effective cloud radius
      total column SO2 
      total column DMS
      Netflux 
    
      Northern Hemisphere Extra-tropical and Tropical Mean Sea Level Pressure difference from global average
    """
                                     )
    parser.add_argument("CONFIG", help="The Name of the Config file")
    parser.add_argument("-d", "--dir", help="The Name of the input directory")
    parser.add_argument("OUTPUT", nargs='?', default=None,
                        help="The name of the output file. Will override what is in the config file")
    parser.add_argument("--clean", help="Clean dumps from directory", action='store_true')
    parser.add_argument("-v", "--verbose", help="Provide verbose output", action="count", default=0)
    args = parser.parse_args()  # and parse the arguments
    # setup processing

    config = StudyConfig.readConfig(expand(args.CONFIG))
    options = config.getv('postProcess', {})

    # work out the files if needed
    if args.dir is None:
        path = options.get('dir', 'share/data/History_Data/')  # path relative to model running dir.
        rootdir = pathlib.Path.cwd() / path
    else:
        rootdir = pathlib.Path(args.dir)
    file_pattern = options.get("file_pattern", '*a.py*.pp')  # file pattern to use.
    files = list(rootdir.glob(file_pattern))

    start_time = options.get('start_time')  # will use None to select.
    end_time = options.get('end_time')
    land_mask_fraction = options.get("land_mask_fraction",0.5)

    clean_files = []
    if args.clean:
        clean_files = rootdir.glob("*a.d*_00")  # pattern for dumps
        extra_files = rootdir.glob("*a.p[4,5,a,d,e,h,k,u,v]*") # all the dump headers  generated. No idea why!
        clean_files += extra_files

    if args.OUTPUT is None:
        output_file = options['outputPath']  # better be defined so throw error if not
    else:
        output_file = args.OUTPUT

    output_file = expand(output_file)

    verbose = args.verbose

    if verbose:  # print out some helpful information...
        print("dir", rootdir)
        print("start_time", start_time)
        print("end_file", end_time)
        print("output", output_file)
        print("clean_files", clean_files)
        print("file_pattern", file_pattern)
        if verbose > 1:
            print("options are ", options)

    results = compute_values(files, output_file, start_time=start_time, end_time=end_time,
                             land_mask_fraction=land_mask_fraction)


    if verbose:  # print out the summary data for all created values
        for name, value in results.items():
            print(f"{name}: {value:.4g}")
        print("============================================================")

    # and possibly clean the dumps
    for file in clean_files:
        if file.samefile(output_file) or file.suffix == '.pp' or file.suffix == 'nc':
            logging.warning(f"Asked to delete {file} but either output file, pp or netcdf so not.")
            continue

        logging.warning(f"Deleting {file}")
        time.sleep(2.)
        file.unlink()  # remove it.


# TODO add  some test cases...

import unittest


class testComp_obs_xarray(unittest.TestCase):
    """
    Test cases for comp_obs_xarray.

    Some cases to try:
    1) That means fn works -- set to 1 > 30N; 2 for lat beteen 0S & 30N; 3 for less than 30S.
    2) That global mean value is close (5%) to the simple mean but not the same...
    3) That for LAT & LPrecip values have the expected number of points..  (basically we are missing Ant. & land only)

    """

    def setUp(self):
        """
        Standard setup for all test cases
        :return: nada
        """


if __name__ == "__main__":
    do_work()
