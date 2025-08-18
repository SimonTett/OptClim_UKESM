#!/usr/bin/env python

""" 

Compute simulated observables using iris (to read in pp datata) and xarray 
 =to process. 
Observations generated are in genProcess (or help for script)

Provides two pathways for date retrieval
1) If input PP data then read each file converting to netcdf. If netcdf already exists then warn and skip read. 
2) If netcdf file(s) open with xarray.open_mfdataset and work with the data.

For processing provide two pathways
1) If time range provided then AFTER processing select within time range and mean in time dropping time dim.
2) If no time range set then leave data as is.

For output provide two pathways
1) Writing to json -- convert to pandas series
2) Writing to netcdf -- write out! 

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
import iris.fileformats.um as iris_um
import logging
import sys
import UKESMlib

import warnings # so we can supress iris warnings...


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning,
                        message='invalid value encountered in divide')
# liust of wanted stash
wanted_stash=set([
                  'm01s02i330', # COSP weighting
                  'm01s02i451','m01s02i452','m01s02i453', # tot cld, liq cld, ice_cld
                  'm01s02i463', 'm01s02i464', # Reff liq & ice.
                  'm01s02i465', # CTP
                  'm01s03i236', # 1.5m temp
                  'm01s01i207','m01s01i208','m01s01i209', # incoming & outgoing SW  and clear sky outgoing
                  'm01s02i205', 'm01s02i206', # Outgoing LW all and clear sky
                  'm01s02i464',
                  'm01s02i285','m01s02i300','m01s02i301','m01s02i302','m01s02i303' # AOD diagnostics
                  'm01s05i063',
                  'm01s50i063', # dry mass of air
                  'm01s34i073','m01s34i102','m01s34i104','m01s34i108','m01s34i11','m01s34i114', # mmr's
                  'm01s34i072', # SO2 MMR
                  'm01s34i071', # DMS
                  'm01s16i222', # MSLP
                  'm01s30i204', # T on P levels
                  'm01s30i206', # RH on P levels
                  'm01s30i301', # heavyside fn
                  'm01s05i216', # sfc precip
              ])

_name_pat = None

my_logger=UKESMlib.my_logger
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
    Uniqueness decided by attributes.
    """
    global _named_vars
    if clear:
        _named_vars = dict()  # reset the global var
    name: str = var.name.replace(" ", "_")
    # convert  all spaces to "_" to make subsequent work easier

    while True:
        value = _named_vars.get(name)
        if value is None:  # not got this value.
            _named_vars[name] = var  # store the variable attributes
            return name  # just return the name
        elif var.equals(value):  # Have the same attrs so OK to use.
            return name
        else:  # generate a new name and continue looping
            name = new_name(name)



    
    # and to extract the 1h data
    # cubes_sub=[c for c in cubes if c.cell_methods[0].intervals[0].startswith('1 hour')]

def read_UMfiles(files: typing.Iterable[pathlib.Path]) -> xarray.Dataset:
    """
    Read a bunch of UM  pp files and convert them to  a dataset
    :param files: iterable of files to read
    :return: dataset
    """
    data_array_list = []
    if len(files) == 0:
        my_logger.warning("File list is empty")
        return
    # check they are all .pp files
        
    my_logger.info(f'Loading iris cubes from {files}')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=FutureWarning)
        cubes = UKESMlib.um_cubes(files,stash_codes=wanted_stash,intervals=[1,6])
    my_logger.info(f'Loaded {len(cubes)}iris cubes')
    for cube in cubes:
        my_logger.debug(f'Read cube: {cube.name()}')
        da = xarray.DataArray.from_iris(cube).rename(cube.name())
        try:
            da = da.expand_dims(dim='time')  # attemp to expand time. Will fail if time already exists
        except ValueError:
            pass
        data_array_list.append(da)
    dataSet = merge_dataArray(data_array_list)
    dataSet.attrs['Conversion'] = 'read_UMfiles'
    return dataSet


def merge_dataArray(dataArray_list: list[xarray.DataArray]) -> xarray.Dataset:
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
            my_logger.info(f"renaming  using {rename_dict}")  # log!
            da = da.rename(rename_dict)
        cleaned.append(da)  # rename

    dataset = xarray.merge(cleaned, compat='override')
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


# def guess_lat_lon_vert_names(dataArray):
#     """
#     Guess the names of latitude, longitude & vertical co-ords in the dataArray
#     starts with latitude/longitude, then lat/lon, then latitude_N/longitude_N then lat_N,lon_N
#     and with atmosphere_hybrid_sigma_pressure_coordinate, altitude, air_pressure for vertical
#
#     :param dataArray:
#     :return: latitude, longitude & vertical co-ord names
#     """
#
#     def find_name(dimensions, patterns):
#         name = None
#         for pattern in patterns:
#             reg = re.compile("^" + pattern + r"(_\d+)?" + "$")
#             for d in dimensions:
#                 if reg.match(d):
#                     name = d
#                     break
#         return name
#
#     dims = dataArray.dims
#     lat_patterns = ['latitude', 'lat']
#     lon_patterns = ['longitude', 'lon']
#     vert_patterns = ['atmosphere_hybrid_sigma_pressure_coordinate',
#                      'altitude', 'pressure', 'air_pressure',
#                      'model_level_number']
#     lat_name = find_name(dims, lat_patterns)
#     lon_name = find_name(dims, lon_patterns)
#     vert_name = find_name(dims, vert_patterns)
#
#     return lat_name, lon_name, vert_name


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
                 scale: typing.Optional[float] = None) -> xarray.DataArray:
    """
    Compute total column of substance (kg/m^2) 
    :param data: dataArray of the variable for which column calculation is being done. (or None)
        Assumed to be a mass mixing ratio (kg/kg)
    :param atmos_mass: total mass/m^2 of atmosphere.
    :param scale:  scale factor.If None no scaling is done
    :return: total_column of substance
    """
    if data is None or atmos_mass is None:
        return None
    lon, lat, vertical_coord,_ = UKESMlib.guess_coordinate_names(data)
    if vertical_coord is None:
        my_logger.warning(f'Failed to find vertical coord in {da.dims}')
        return None
    col = (data * atmos_mass).sum(vertical_coord)  # work out column  by integrating and multiply by atmos/m^2
    col = col.rename('Column '+data.name )
    
    if scale is not None:
        col *= scale

    return col


def names(dataset:xarray.Dataset, name=None):
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
        # key depends on how it was converted.
        if dataset.attrs.get('Conversion','') == 'read_UMfiles':
            key = 'STASH'
        else:
            key='um_stash_source'
    else:
        raise ValueError(f"Do not know what to do with {name}")

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




def genProcess(dataset:xarray.Dataset,
               exclude_vars:typing.Optional[list[str]]=None) -> typing.Dict[str,xarray.DataArray]:
    """
    Setup the processing information
    :param dataset: the dataset containing the data
    :return: dict containing data to be processed.
    """
    # create long & standard name lookup
    lookup_std = names(dataset)
    lookup_long = names(dataset, name='long')
    lookup_stash = names(dataset, name='stash')
    pseudolev_550nm = 3 # pseudolevel for 550 nm in UKESM1.1

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
            my_logger.warning(f"Failed to find name {name} of type {name_type}")
            return None
        da = dataset[var]
        if (len(args) > 0) or (len(kwargs) > 0):
            try:
                da = da.sel(*args, **kwargs)

            except KeyError:  # failed to find some co-ords
                my_logger.warning(f"Failed to select {var} using {args} or {kwargs}")
                return None
        return da

    def modis_fn(stash,dataset,name,scale=None):
        # codes are
        # 'm01s02i463' liquid Reff
        # 'm01s02i464' ice  Reff
        # 'm01s02i451'  cloud fraction
        # 'm01s02i452' liquid cloud fraction
        # 'm01s02i453' ice cloud fraction
        # 'm01s02i465' COSP MODIS ctp
        modis_wt = name_fn('m01s02i330',dataset,name_type='stash') # COSP MODIS weight
        modis_wt_value = name_fn(stash,dataset,name_type='stash') # COSP cld weighted value
        if (modis_wt_value is None) or (modis_wt is None):
            my_logger.warning(f"Failed to find modis_wt_value or modis_wt for {name} using stash {stash}")
            return None
        result = modis_wt_value.where(modis_wt > 1e-5) / modis_wt  # compute the effective radius
        result = result.rename(name)  # rename the result
        if scale is not None:
            result = result * scale
        return result
        


    def AOD(dataset:xarray.Dataset,
            *args,
            **kwargs) -> typing.Optional[xarray.DataArray]:
        """
        Compute the Aerosol Optical Depth (AOD) at 550 nm. Uses the following stash codes (and names):
        2285 -- dust (from CLASSIC(?) scheme)
        2300 -- Aitkin (soluble) AOD
        2301 -- Accumulation (soluble) abs optical depth
        2302 -- Coarse (soluble) abs optical depth
        2303 -- Aitkin (insoluble) abs optical depth
        
        :param dataset: dataset containing the data
        :return: AOD as a DataArray or None if any of the components are not found.
    """
        ##2585 -- Mineral dust optical depth in radiation scheme
        items = [285,300,301,302,303]  # stash item codes for AOD
        # From Jane Mulchay -- To calculate the total AOD in UKESM1.1 you sum up the following stashcodes:
        # 2300+2301+2302+2303+2285
        result = 0.0 # initialise result
        missing = []
        for item in items:
            stash = f'm01s02i{item:03d}'
            da = name_fn(stash,dataset,name_type='stash',*args,**kwargs)
            if da is None:  # if not found then skip
                missing += [stash]  # add to missing list
            else:
                result += da.sel(pseudo_level=pseudolev_550nm)  # get the dataArray for the item and  add to result
        if len(missing) > 0:  # if any missing then return None
            my_logger.warning(f"Failed to find AOD components {missing}")
            return None
        result = result.rename('AOD')  # rename the result
        return result

    def cell_area(da):
        # compute area on m^2 useing haversine formulae
        # Earth's radius in meters
        R = 6371e3
        # Convert degrees to radians
        lats_rad = np.deg2rad(da['latitude'])
        lons_rad = np.deg2rad(da['longitude'])
        # Calculate spacing
        dlat = np.abs(np.diff(lats_rad).mean())
        dlon = np.abs(np.diff(lons_rad).mean())
        # Area formula for each latitude band
        area = (R**2) * dlon * (np.sin(lats_rad + dlat/2) - np.sin(lats_rad - dlat/2))
             
        return area  # 1D array: area for each latitude band



    lon,latitude_coord, vert,tc = UKESMlib.guess_coordinate_names(dataset[lookup_stash['m01s03i236']])  # 1.5 m air tempo.
    constrain_60S = dataset[latitude_coord] >= -60.
    # need to extract the actual values...
    constrain_60S = constrain_60S[latitude_coord][constrain_60S]
    
    # set up the data to be meaned. Because of xarray's use of dask no processed  happens till
    # spatial (And temporal ) means computed. See means.
    mass = name_fn('m01s50i063', dataset, name_type='stash')
    if mass is not None: # convert to mass/m^2
        area = cell_area(mass)
        mass /= area # convert to per m^2
    AOD_range = slice(-55,55) # want between +/- 55). Hopefully enough sunshine there
    land_range = slice(-60,None) 
    process = {
        'OLR': name_fn('toa_outgoing_longwave_flux', dataset, name_type='standard'),
        'OLRC': name_fn('toa_outgoing_longwave_flux_assuming_clear_sky', dataset, name_type='standard'),
        'RSR': name_fn('toa_outgoing_shortwave_flux', dataset, name_type='standard'),
        'RSRC': name_fn('toa_outgoing_shortwave_flux_assuming_clear_sky', dataset, name_type='standard'),
        'INSW': name_fn('toa_incoming_shortwave_flux', dataset, name_type='standard'),
        'T2m':name_fn('m01s03i236',dataset,name_type='stash',latitude=land_range),
        'Precip': name_fn('precipitation_flux',dataset,name_type='standard',
                          latitude=land_range),
        'MSLP': name_fn('air_pressure_at_sea_level',dataset,name_type='standard'),
        'Reff': modis_fn('m01s02i463',dataset,'Reff'),
        'ReffIce': modis_fn('m01s02i464',dataset,'ReffIce'),
        'CLDtot': modis_fn('m01s02i451',dataset,'CLDtot'),
        'CLDliq': modis_fn('m01s02i452',dataset,'CLDliq'),
        'CLDice': modis_fn('m01s02i453',dataset,'CLDice'),
        'CTP': modis_fn('m01s02i465',dataset,'CTP',scale=1e-2),
        'AOD_550': AOD(dataset,latitude=AOD_range),
        'Dust_AOD_550': name_fn('m01s02i285',dataset,name_type='stash',pseudo_level=pseudolev_550nm,latitide=AOD_range),
        # SO2 related.
        'SO2_col': total_column(name_fn('mass_fraction_of_sulfur_dioxide_in_air', dataset, name_type='name'),
                                mass),
        # UM/UKCA does have total column diagnostics for H2SO4 but they are not in the UKESM1.1 diagnostics.
        'Trop_SW_up': name_fn('tropopause_upwelling_shortwave_flux', dataset),
        'Trop_SW_net': name_fn('tropopause_net_downward_shortwave_flux', dataset),
        'Trop_LW_up': name_fn('tropopause_upwelling_longwave_flux', dataset),
        'Trop_LW_net': name_fn('tropopause_net_downward_longwave_flux', dataset),

    }
    # add in total columsn
    col_names = ['h2so4','nucleation_h2so4','aitkin_h2so4','accum_h2so4','coarse_h2so4','DMS']
    stash_codes = ['m01s34i073','m01s34i102','m01s34i104','m01s34i108','m01s34i114','m01s34i071']
    
    for name,stash in zip(col_names,stash_codes):
        process[name+'_col']= total_column(name_fn(stash,dataset,name_type='stash'), mass) # FIXME -- values are -ve

    # deal with T and rh values
    coord_500hPa = dict(pressure=500)  # co-ord value for 500 hPa
    coord_50hPa = dict(pressure=50)  # co-ord value for 50 hPa

    p_wt = name_fn('Heavyside function on pressure levels', dataset, name_type='long')
    if p_wt is not None: # got the p_wts
        temp = name_fn('m01s30i204', dataset, name_type='stash')
        if temp is not None:
            temp /= p_wt # scale by time above sfc.
            process.update(
                {'T@50': temp.sel(coord_50hPa).rename('T@50'),
                 'T@500': temp.sel(coord_500hPa).rename('T@500')})

        # compute RH
        rh = name_fn('m01s30i206', dataset, name_type='stash')
        if rh is not None:
            rh /= p_wt # scale by time above sfc
            process.update(
                {'RH@500': rh.sel(coord_500hPa).rename('RH@500'),
                 'RH@50': rh.sel(coord_50hPa).rename('RH@50'), })


            
    process['netflux'] = (process['INSW'] - process['RSR'] - process['OLR']).rename('netflux')
    # now remove excluded values. As using dask then processing does not actually happen till later!
    # consider extending to use regexps
    for var in exclude_vars:
        process[var]=None # just set the variable to None and then it won't get processed
        my_logger.debug(f'Excluded {var}')

    return process



time_range_type = tuple[typing.Optional[str],typing.Optional[str]]
def compute_values(files: typing.Iterable[pathlib.Path],
                   land_mask:xarray.DataArray,
                   time_range:typing.Optional[time_range_type] = None,
                   exclude_vars:typing.Optional[list[str]] = None,
                   land_mask_fraction:float = 0.5) -> xarray.Dataset:
    """

    :param files: iterable of files to readin
    :param output_file: output file to write to
    :param start_time: start time as iso date/time
    :param end_time:  end_time as iso date/time
    :return: dict of results
    """
    if all([file.suffix == '.pp' for file in files]):
        my_logger.debug(f'Reading pp data from {len(files)} files')
        dataset = read_UMfiles(files)
    elif all([file.suffix in ['.nc','.hdf'] for file in files]):
        my_logger.info(f'Reading netcdf data from {len(files)} files')
        dataset=[]
        for file in sorted(files):
            dataset.append(xarray.open_dataset(file))
            my_logger.debug(f"Opened  {file}")
        dataset = xarray.concat(dataset,dim='time')
                      
        my_logger.info(f'Read netcdf data from {len(files)} files')

    else:
        raise ValueError(f'Files inconsisent - {files}')
    if time_range is not None:
        time=" ".join(dataset.time.dt.strftime('%Y-%m-%d').values)
        dataset = dataset.sel(time=slice(*time_range))
        if len(dataset.time) == 0:
            raise ValueError(f'No data in time range {time_range} for times: {time}')
        else:
            my_logger.info(f'Procesing data for {len(dataset.time)} times')

    masks = UKESMlib.create_region_masks(land_mask,critical_value=land_mask_fraction)
    process = genProcess(dataset,exclude_vars=exclude_vars)

    # now to process all the data making output.
    results=dict()
    for name, dataArray in process.items():
        if dataArray is None:  # no dataarray for this name
            my_logger.warning(f"{name} is None. Not processing")
            continue
        da = dataArray.squeeze(drop=True).reset_coords(drop=True)
        lon,lat,_,_ = UKESMlib.guess_coordinate_names(da)
        da = da.rename({lon:'longitude',lat:'latitude'}) # stadnardise co-ord names
        regrid = UKESMlib.conservative_regrid(da,land_mask)
        mean = UKESMlib.da_regional_avg(regrid, masks)
        result = UKESMlib.process(mean).compute()
        results[name] = result
        my_logger.info(f'Processed {name}')
    
    results = xarray.Dataset(results)
    

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
    ## code from cpt4-1 with prompt to track which arguments were set
    class StoreWithFlag(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            #if values is None:
            #    vaalues = parser.get_default(self.dest)
                
            setattr(namespace,self.dest,values)
            setattr(namespace, f"_{self.dest}_set", True)
    
    class StoreWithFlag_store_true(argparse._StoreTrueAction):
        def __call__(self, parser, namespace, values, option_string=None):
            #setattr(namespace,self.dest,values)
            super().__call__(parser, namespace, values, option_string)
            setattr(namespace, f"_{self.dest}_set", True)

    class StoreWithFlag_count(argparse._CountAction):
        def __call__(self, parser, namespace, values, option_string=None):
            super().__call__(parser, namespace, values, option_string)
            setattr(namespace, f"_{self.dest}_set", True)

    class StoreWithFlag_BooleanOptionalAction(argparse.BooleanOptionalAction):
        def __call__(self, parser, namespace, values, option_string=None):
            super().__call__(parser, namespace, values, option_string)
            setattr(namespace, f"_{self.dest}_set", True)

    
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
    defaults = dict(dir='share/data/History_Data/',
                          output_file='observations.json',
                          clean=False,
                          verbose=0,
                          log_level = 'WARNING',
                          mask_variable='field36',
                          land_mask_fraction=0.5,
                          file_pattern = '*a.pm*.pp',
                          exclude_vars=[],
                          timeseries=False,
                          overwrite=True
                          
                          )
    parser.add_argument("CONFIG", help="The Name of the Config file",type=pathlib.Path)
    parser.add_argument("-d", "--dir",
                        help="The Name of the input directory",type=pathlib.Path)
    parser.add_argument("output_file",  nargs='?',type=pathlib.Path,
                        help="The name of the output file.")
    parser.add_argument("--clean", help="Clean dumps from directory", action=argparse.BooleanOptionalAction)
    parser.add_argument("-v", "--verbose", help="Provide verbose output", default=0,action='count')
    parser.add_argument('--log_level',help='Set logging level',default='WARNING')
    parser.add_argument('--time_range',help='Time range for data extraction',nargs=2)
    parser.add_argument('--mask_file',help='Name of file containing land_mask data',type=pathlib.Path)
    parser.add_argument('--mask_variable',help='Name of land mask variable')
    parser.add_argument('--land_mask_fraction',help='Critical value for land',type=float)
    parser.add_argument('--file_pattern',help='File glob pattern to read data from')
    parser.add_argument('--exclude_vars',help='Things not to process',nargs='+')
    parser.add_argument('--timeseries',help='Have timeseries output',action=argparse.BooleanOptionalAction)
    parser.add_argument('--overwrite',help='Overwrite',action=argparse.BooleanOptionalAction)
    args = parser.parse_args()  # and parse the arguments
    
    # setup processing
    with args.CONFIG.open('rt') as fp:
        json_options = json.load(fp)  # load the options from the config file

    options=defaults.copy()
    options.update({k:v for k,v in json_options.items() if (v is not None and not k.endswith('_comment'))})
    # update with json values. Ignoring any Nones we get
    
    
    # overwrite the options with arg values if they were set
    options.update({
        arg_name:value for   arg_name,value in vars(args).items() if value is not None})
    # set types appropriately -- based on args!
    # Converting everything to the correct type.
    for act in parser._actions: # hack using private part of parser
        if act.type is not None:
            try:
                options[act.dest]=act.type(options[act.dest])
            except KeyError:
                my_logger.debug(f'Did not find {act.dest} so no type conversion')


    # set up the logger.
    UKESMlib.setup_logging(options['log_level'])
            
    # write out the options to logger
    for k,v in options.items():
        my_logger.info(f'Option[{k}] = {v}')

    # extract the options
    rootdir = pathlib.Path(options['dir'])
    file_pattern = options["file_pattern"]
    time_range = options.get('time_range')
    land_mask_file = expand(options['mask_file'])
    land_mask_var = options['mask_variable']
    land_mask_fraction = options["land_mask_fraction"]
    timeseries = options['timeseries']
    clean = options['clean']
    output_file = options['output_file']  
    verbose=options['verbose']
    overwrite=options['overwrite']
    exclude_vars=options['exclude_vars']

    if output_file.exists() and not overwrite:
        raise ValueError(f'Output file {output_file} exists and overwrite set to false')

    files = list(rootdir.glob(file_pattern)) # files to process
    if len(files) ==0:
        ValueError("Failed to find any files. Exiting")
    my_logger.info(f'Processing {len(files)}')


    # Handle cleaing 
    clean_files = []
    if clean:
        clean_files = list(rootdir.glob("*a.d*_00"))  # pattern for dumps
        extra_files = list(rootdir.glob("*a.p[4,5,a,b,c,d,e,f,g,h,i,j,k,u,v]*")) # all the dump headers  generated. No idea why!
        clean_files += extra_files

    land_mask = xarray.load_dataset(land_mask_file,decode_times=False)[land_mask_var]
    land_mask = land_mask.squeeze(drop=True)

    # all setup so can now compute the summary values.
    results = compute_values(files, land_mask,
                             time_range=time_range,
                             land_mask_fraction=land_mask_fraction,
                             exclude_vars=exclude_vars)


    if not timeseries: # time average unless we are a timeseries.
        for k in results.keys():
            results[k]=results[k].mean('time')

    # write the data out
    if output_file.suffix == '.json': # json file
        # Want to flatten across variable for json
        r2=dict()
        for variable in results.keys():
            series = results[variable].to_series()
            r2.update(
                {f'{variable}_{k}':float(v) for k,v in series.items()}
            )
        
        with output_file.open('wt') as fp:
            json.dump(r2, fp, indent=2)
    elif output_file.suffix == '.nc': # just write it out for netcdf.
        results.to_netcdf(output_file,unlimited_dims=['time'])
    else:
        raise ValueError(f'Do not know what to do with {output_file}')
    
    # and possibly clean the dumps -- probably not needed
    if len(clean_files) > 0:
        logging.warning(f"Deleting {len(clean_files)} files. Sleeping 10")
        time.sleep(10)
    for file in clean_files:
        if file.samefile(output_file) or file.suffix == '.pp' or file.suffix == 'nc':
            logging.warning(f"Asked to delete {file} but either output file, pp or netcdf so not doing so.")
            continue

        logging.debug(f"Deleting {file}")
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
