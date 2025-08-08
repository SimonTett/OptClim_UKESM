import xarray 
import typing
import logging
import numpy as np
import socket
import pathlib
import os 
logger = logging.getLogger(__name__)

## work out base-dirs for data depening on machine
host = socket.gethostname()
try:
    base_dir = pathlib.Path(os.getenv('BASE_DIR'))
except TypeError as e: # failed coz BASE_DIR does not exist
    logger.warning('BASE_DIR not in env')
    if host.startswith('GEOS-W'):  # Geos windows dekstop
        base_dir = pathlib.Path(r"P:\optclim_data")
    else:
        raise ValueError('Do not know how to define base_dir.  Define BASE_DIR or modify code')



def guess_coordinate_names(da: xarray.DataArray) -> \
        tuple[typing.Optional[str],typing.Optional[str],typing.Optional[str],typing.Optional[str]]:
    possible_lon_names = ['longitude', 'lon', 'x', 'long']
    possible_lat_names = ['latitude', 'lat', 'y']
    possible_time_names = ['time', 't', 'date', 'valid_time']
    possible_vert_names = ['atmosphere_hybrid_sigma_pressure_coordinate',
                     'altitude', 'pressure', 'air_pressure', 'depth','level','z','Z'
                     'model_level_number']
    
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
    logger.info("Running conservative regridding...")
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
    logger.info("Creating regional masks...")
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
    logger.info("Using cos(lat) weights...")
    lon_name, lat_name, _,_ = guess_coordinate_names(da)
    lat = da[lat_name]

    weights = np.cos(np.deg2rad(lat))
    return weights


def compute_regional_averages(ds: xarray.Dataset,
                              masks: dict[str, xarray.DataArray],
                              ) -> xarray.Dataset:
    logger.info("Computing regional averages...")
    results = {}
    area_weights = compute_area_weights(ds)

    lon_name, lat_name, _,_ = guess_coordinate_names(ds)
    for var_name, var_data in ds.data_vars.items():
        if not is_lon_lat(var_data):
            logging.info(f'{var_name} is not a long/lat field. Skipping')
            continue

        logging.info(f'Processing {var_name}')

        result = []
        for region_name, mask in masks.items():
            masked_var = var_data.where(mask)
            mn = masked_var.weighted(area_weights).mean(dim=[lon_name, lat_name], skipna=True)
            mn = mn.expand_dims(region=[region_name])
            mn.compute()  # force dask to compute this.
            result.append(mn)
        results[var_name] = xarray.concat(result, dim='region')

    return xarray.Dataset(results)
