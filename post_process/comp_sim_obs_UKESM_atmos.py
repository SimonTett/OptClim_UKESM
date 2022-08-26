#!/usr/bin/env python

""" Compute simulated observables using xarray. 
xarray appears to be a lot faster than iris!
 Based on comp_obs. Hopefully faster 
Observations generated are in genProcess (or help for script). This version set up for UKESM1 and direct netcdf output
from model.

 """

import argparse  # parse arguments
import json  # get JSON library
import os  # OS support
import pathlib
import re

import numpy as np
import xarray

import StudyConfig
lookup_std=dict() # where we hold std names lookup
lookup_long = dict() # where we hold long names lookup

def guess_lat_lon_vert_names(dataArray):
    """
    Guess the names of latitude,longitude & vertical co-ords in the dataArray
    starts with latitude/longitude, then lat/lon, then latitude_N/longitude_N then lat_N,lon_N
    and with atmosphere_hybrid_sigma_pressure_coordinate, altitude, air_pressure for vertical

    :param dataArray:
    :return: latitude, longitude & vertical co-ord names
    """
    if dataArray is None:
        return None, None, None

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
    vert_patterns = ['atmosphere_hybrid_sigma_pressure_coordinate', 'altitude', 'air_pressure']
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


def total_column(data, delta_p, vertical_coord=None):
    """
    Compute total column of substance (kg/m^2) assuming hydrostatic atmosphere.
        (Atmospheric mass in layer is \Delta P/g)
    :param data: dataArray of the variable for which column calculation is being done.
        Assumed to be a mass mixing ratio (kg/kg)
    :param delta_p: dataArray of pressure thicknesses (Pa) for each level.
    :param vertical_coord: default is None. If None then will be guessed using guess_lat_long_vert_names()
    :return: total_column of substance
    """

    lat, lon, vertical_coord = guess_lat_lon_vert_names(data)
    mass = (data * delta_p / 9.81).sum(vertical_coord)  # work out mass by integrating.

    return mass


def names(dataset, name=None):
    """
     Return dictionary of standard (or long)  names for each variable in a dataset.

    :param dataset: xarray data
    :param name: what you want to return (None, 'standard','long'). If None (default) then standard_name will be returned
    :return: dict of standard names
    """

    if name is None or name == 'standard':
        key = 'standard_name'
    elif name == 'long':
        key = 'long_name'
    else:
        raise Exception(f"Do not know what to do with {name}")

    lookup = {}
    for var in dataset.variables:
        try:
            name = dataset[var].attrs[key]
            lookup[name] = var  # if not present then won't update lookup
        except KeyError:
            pass

    return lookup





def genProcess(dataset, land_mask, latitude_coord=None):
    """
    Setup the processing information
    :param dataset: the dataset containing the data
    :param land_mask -- land mask as a dateArray.
    :param latitude_coord -- name of latitude co-ord (optional; default None)
      If set to None then will be guessed using
    :return: dict containing data to be processed.
    """
    # create long & standard name lookup
    global  lookup_std
    global lookup_long
    lookup_std = names(dataset)

    lookup_long = names(dataset, name='long')

    if latitude_coord is None:
        latitude_coord, lon, vert = guess_lat_lon_vert_names(dataset[lookup_long['TEMPERATURE AT 1.5M']])
    constrain_60S = dataset[latitude_coord] >= -60.
    # need to extract the actual values...
    constrain_60S = constrain_60S.latitude[constrain_60S]

    coord_500hPa = dict(air_pressure=500)  # co-ord value for 500 hPa
    coord_50hPa = dict(air_pressure=50)  # co-ord value for 50# hPa
    # set up the data to be meaned. Because of xarray's use of dask no loading happens till
    # data actually processed (below)
    # delta_p = model_delta_p(dataset.surface_air_pressure,
    #                         dataset.UM_atmosphere_hybrid_sigma_pressure_coordinate_ak_bounds,
    #                         dataset.UM_atmosphere_hybrid_sigma_pressure_coordinate_bk_bounds)
    delta_p = None  # not bothering to set up delta P as not actually used.

    process = {
        # 'TEMP@50': dataset[lookup_long['TEMPERATURE ON P LEV/T GRID']].sel(coord_50hPa, method='nearest'),
        'TEMP@500': extract_ds(dataset, 'TEMPERATURE ON P LEV/T GRID', coord=coord_500hPa),
        'RH@500': extract_ds(dataset, 'RELATIVE HUMIDITY ON P LEV/T GRID', coord=coord_500hPa),
        'OLR': extract_ds(dataset, 'toa_outgoing_longwave_flux'),
        'OLRC': extract_ds(dataset, 'toa_outgoing_longwave_flux_assuming_clear_sky'),
        'RSR': extract_ds(dataset, 'toa_outgoing_shortwave_flux'),
        'RSRC': extract_ds(dataset, 'toa_outgoing_shortwave_flux_assuming_clear_sky'),
        'INSW': extract_ds(dataset, 'toa_incoming_shortwave_flux'),
        'LAT': extract_ds(dataset, 'TEMPERATURE AT 1.5M', coord={latitude_coord: constrain_60S}, mask=land_mask),
        'Lprecip': extract_ds(dataset, 'precipitation_flux', coord={latitude_coord: constrain_60S}, mask=land_mask),
        'MSLP': extract_ds(dataset, 'air_pressure_at_sea_level'),
        # 'REFF': dataset['UM_m01s01i245'] / dataset['UM_m01s01i246'],
        # HadXM3 carries all S compounds as mass of S. So SO2 & DMS need conversion to SO2 & DMS for
        # comparison with obs. Code below will need updating for HadGEM-GA10?
        # 'SO2_col': total_column(dataset.mass_fraction_of_sulfur_dioxide_in_air, delta_p) * \
        #            simonLib.mole_wt['SO2'] / simonLib.mole_wt['S'],
        # 'aitkin_col': total_column(dataset[lookup_long['SO4 AITKEN MODE AEROSOL AFTER TSTEP']], delta_p),
        # 'accum_col': total_column(dataset[lookup_long['SO4 ACCUM. MODE AEROSOL AFTER TSTEP']], delta_p),
        # 'DMS_col': total_column(dataset.mass_fraction_of_dimethyl_sulfide_in_air, delta_p) * \
        #            simonLib.mole_wt['DMS'] / simonLib.mole_wt['S'],
        # 'O3_col_DU': total_column(dataset['UM_m01s02i260'], delta_p) * simonLib.DU['O3'],
        # 'Trop_SW_up': dataset.tropopause_upwelling_shortwave_flux,
        # 'Trop_SW_net': dataset.tropopause_net_downward_shortwave_flux
    }

    process['netflux'] = process['INSW'] - process['RSR'] - process['OLR']

    return process


def extract_ds(dataset, name, coord=None, mask=None):
    var = lookup_std.get(name, lookup_long.get(name))  # try standard name first. If not found use long name
    if var is None:
        print(f"Failed to find {name} in standard or long list ")
        return None  # no name found so return empty

    da = dataset[var]

    if mask is not None:
        da = da.where(mask, np.nan)
    if coord is not None:
        da = da.sel(coord, method='nearest')
    return da
def means(dataArray, name, latitude_coord=None):
    """ 
    Compute means for NH extra tropics, Tropics and SH extra Tropics. 
    Tropics is 30N to 30S. NH extra tropics 30N to 90N and SH extra tropics 90S to 30S 
    Arguments:
        :param dataArray -- dataArray to be processed. If None a bunch of Nones will be returned
        :param name -- name to call mean.
        :param latitude_coord: name of the latitude co-ord. Default is NOne.
            If not set then guess_lat_long_vert_names() will be used


    """

    if latitude_coord is None:
        latitude_coord, lon, vert = guess_lat_lon_vert_names(dataArray)

    if dataArray is not None:
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
        if dataArray is None:
            means[name + '_' + rgn_name] = None
        else:
            if rgn_fn is None:
                v = dataArray.squeeze().weighted(wt).mean()
            else:
                msk = rgn_fn(dataArray[latitude_coord])  # T where want data
                v = dataArray.where(msk, np.nan).squeeze().weighted(wt.where(msk, 0.0)).mean()
            means[name + '_' + rgn_name] = float(v.load().squeeze().values)
            # store the actual values -- losing all meta-data

    return means  # means are what we want


def do_work():
    # parse input arguments

    parser = argparse.ArgumentParser(description="""
    Post process Unified Model data to provide  simulated observations. Example use is:
    
    comp_sim_obs.py input.json --output output.json --dir /work/n02/n02/tetts/cylc-run/u-cp254/share/data/History_Data/
    
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
      Netflux 
      Northern Hemisphere Extra-tropical and Tropical Mean Sea Level Pressure difference from global average
    """
                                     )
    parser.add_argument("CONFIG", help="The Name of the Config file")
    parser.add_argument("-d", "--dir", help="The Name of the input directory")
    parser.add_argument("output", help="The name of the output file. Will override what is in the config file")
    parser.add_argument("-v", "--verbose", help="Provide verbose output", action="count", default=0)
    args = parser.parse_args()  # and parse the arguments
    # setup processing

    config = StudyConfig.readConfig(args.CONFIG)
    options = config.getv('postProcess', {})

    # work out the files if needed
    if args.dir is None:
        rootdir = pathlib.Path.cwd()
    else:
        rootdir = pathlib.Path(os.path.expandvars(args.dir))
    files = list(rootdir.glob('*_pm*.nc')) # only want monthly mean netcdf files.

    mask_file = options['mask_file']
    mask_file = pathlib.Path(os.path.expandvars(mask_file)).expanduser()
    mask_name = options['mask_name']
    mask_fraction = options.get('mask_fraction')  # if not present will be None
    start_time = options['start_time']
    end_time = options['end_time']
    if args.output is None:
        output_file = options['outputPath']  # better be defined so throw error if not
    else:
        output_file = args.output
    # and deal with any env vars
    output_file = os.path.expandvars(output_file)

    verbose = args.verbose

    if verbose:  # print out some helpful information..
        print("dir", rootdir)
        print("mask_file", mask_file)
        print("land_mask", mask_name)
        print("mask_fraction", mask_fraction)
        print("start_time", start_time)
        print("end_file", end_time)
        print("output", output_file)
        if verbose > 1:
            print("options are ", options)
    # UM ancillary times are usually  year 0 which cases grief if attempt to decode
    land_mask = xarray.load_dataset(mask_file, decode_times=False)[mask_name].squeeze()  # land/sea mask
    if mask_fraction is not None:
        land_mask = land_mask >= mask_fraction
    latitude_coord = options.get('latitude_coord', None)
    dataset = xarray.open_mfdataset(files)
    # need to rename various elements
    rename = dict(TMONMN='time', TMONMN_rad='time_rad',
                  PLEV10='air_pressure',
                  latitude_t='latitude', longitude_t='longitude')
    dataset = dataset.rename(rename)
    # now change time_rad to time.
    # check time_rad values = time values
    if (dataset.time == dataset.time_rad).all():
        dataset = dataset.reset_index(['time_rad'], drop=True)
        dataset = dataset.rename(dict(time_rad='time'))
    dataset = dataset.sortby('time')  # sort by time coords.
    dataset = dataset.sel(time=slice(start_time, end_time))

    process = genProcess(dataset, land_mask, latitude_coord=latitude_coord)

    # now to process all the data making output.
    summary = dict()
    for name, dataArray in process.items():
        mean = means(dataArray, name, latitude_coord=latitude_coord)  # compute the means
        summary.update(mean)  # and stuff them into the results dict.
        if verbose > 1:
            print(f"Processed {name} and got {mean}")

    # now fix the MSLP values. Need to remove the global mean from values and the drop the SHX value.
    # TODO put this in mean calculation as an option..
    summary.pop('MSLP_SHX')
    for k in ['MSLP_NHX', 'MSLP_TROPICS']:
        summary[k + '_DGM'] = summary.pop(k) - summary['MSLP_GLOBAL']

    # lower case all names
    results = dict()
    for k, v in summary.items():
        results[k.lower()] = v
    if verbose:  # print out the summary data for all created values
        for name, value in results.items():
            if value is None:
                print(f"{name.lower()}: {value}")
            else:
                print(f"{name.lower()}: {value:.4g}")
        print("============================================================")

    # now to write the data
    with open(output_file, 'w') as fp:
        json.dump(results, fp, indent=2)

    return results


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
    results = do_work()
