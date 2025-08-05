#!/usr/bin/env python3
"""
Process MODIS and AATSR cloud data to extract relevant variables, regrid, and mask overlapping data.#
MODIS data is extracted from MCD06COSP_M3_MODIS files, while AATSR data is extracted from ESACCI-L3C_CLOUD-CLD_PRODUCTS-AATSR_ENVISAT files.
liquid and ice cloud fractions are computed from AATSR data, and then a subset of the AATSR data is regridded to equivalent MODIS data
"""
import xarray
import xarray_regrid
import pathlib
import logging
import numpy as np
import argparse

my_logger = logging.getLogger(__name__)
if my_logger.hasHandlers():
    my_logger.handlers.clear()
my_logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

my_logger.addHandler(handler)
my_logger.propagate = False  # Prevents double logging if root logger is also configured


def ice_liq_cld_fraction(ds: xarray.Dataset) -> xarray.Dataset:
    """
    Compute liquid and ice cloud fractions from AATSR cloud fraction data.

    Parameters:
    ds: input dataset containing AATSR cloud fraction data.

    Returns:
    dataset: with Liquid cloud fraction, Ice cloud fraction.
    """
    script_name = pathlib.Path(__file__).name
    cld_frac = ds['cfc']
    liq_frac_total = ds['cph']
    liq_cld_frac = (cld_frac * liq_frac_total).assign_attrs(long_name='liquid cloud fraction',
                                                            standard_name='liquid_water_cloud_area_fraction', units='1',
                                                            valid_min=0.0, valid_max=1.0)
    ice_cld_frac = (cld_frac * (1 - liq_frac_total)).assign_attrs(long_name='ice cloud fraction',
                                                                  standard_name='ice_cloud_area_fraction', units='1',
                                                                  valid_min=0.0, valid_max=1.0)
    # set the attributes for the new variables

    result = xarray.Dataset({
        'liquid_cloud_fraction': liq_cld_frac,
        'ice_cloud_fraction': ice_cld_frac,

    })
    result = result.assign_attrs(ds.attrs)
    result.attrs.update(
        history=[f'Extracted liquid and ice cloud fractions from AATSR cloud fraction data using {script_name}.'] + [
            result.attrs.get('history', '')])

    return result


def modis_set_time(ds: xarray.Dataset) -> xarray.Dataset:
    """
    Set the time coordinate for MODIS datasets.

    Parameters:
    ds: input dataset containing MODIS data.

    Returns:
    dataset: with time coordinate set.
    """
    start_time = np.array(ds.attrs['time_coverage_start'], dtype='datetime64')
    end_time = np.array(ds.attrs['time_coverage_end'], dtype='datetime64')

    time_bounds = xarray.DataArray([start_time, end_time], dims=['nbounds', 'time'],
                                   attrs={'long_name': 'time_bounds', 'standard_name': 'time_bounds',
                                          'units': 'days since 1970-01-01'})
    ds_modis = ds.assign_coords(time=start_time)
    ds_modis['time_bounds'] = time_bounds
    return ds_modis


# set up cmd line arguments
base_dir_default = pathlib.Path(r"P:\optclim_data\obs_data")  # do system dependent guess here
parser = argparse.ArgumentParser(description='Process MODIS and AATSR cloud data.')
parser.add_argument('base_dir', type=pathlib.Path, help='Base directory for input and output data')
group = parser.add_mutually_exclusive_group()
group.add_argument('--overwrite', dest='overwrite', action='store_true', help='Enable overwrite')
group.add_argument('--nooverwrite', dest='overwrite', action='store_false', help='Disable overwrite')
parser.set_defaults(overwrite=False)
args = parser.parse_args()

# variables wanted from MODIS and match in AATSR data.
match_dir = {'Cloud_Mask_Fraction':'cfc',
             'Cloud_Top_Pressure': 'ctp',
             'Cloud_Retrieval_Fraction_Liquid': 'liquid_cloud_fraction',
             'Cloud_Retrieval_Fraction_Ice': 'ice_cloud_fraction',
             'Cloud_Retrieval_Fraction_Total': 'cfc',
             'Cloud_Particle_Size_Liquid': 'cer_liq',
             'Cloud_Particle_Size_Ice': 'cer_ice'}
base_dir = args.base_dir
modis_dir = base_dir / "modis_cloud"
aatsr_dir = base_dir / "aatsr_cloud"
extract_modis_dir = base_dir / "modis_cloud_extract"
extract_aatsr_dir = base_dir / "aatsr_cloud_extract"
for dir in [extract_modis_dir, extract_aatsr_dir]:
    dir.mkdir(parents=True, exist_ok=True)  # Create directories if they do not exist

modis_files = list(modis_dir.glob("MCD06COSP_M3_MODIS.A*.062.2022*.nc"))
aatsr_files = list(
    aatsr_dir.glob("*-ESACCI-L3C_CLOUD-CLD_PRODUCTS-AATSR_ENVISAT-fv3.0.nc"))  # just want the AATSR files.
# need the times for MODIS as they are stored as attributes and need to be converted to datetime
time = []
time_bounds = []
modis_attrs = None
for file in modis_files:
    with xarray.open_dataset(file) as ds:
        start_time = np.array(ds.attrs['time_coverage_start'], dtype='datetime64')
        end_time = np.array(ds.attrs['time_coverage_end'], dtype='datetime64')
        time += [start_time]
        time_bounds += [np.array([start_time, end_time], dtype='datetime64')]
        if modis_attrs is None:  # Get attributes from the first file
            modis_attrs = ds.attrs.copy()
            for v in ['time_coverage_start', 'time_coverage_end']:  # attributes we do not want.
                modis_attrs.pop(v, None)

# add in a history entry
modis_attrs['history'] = modis_attrs.pop('history', '') + f' Processed MODIS data using {pathlib.Path(__file__).name}.'

# set up co-ordinates for MODIS data
time = np.array(time)
time_bounds = xarray.DataArray(time_bounds, dims=['time', 'nbounds'],
                               coords={'time': time, 'nbounds': [0, 1]},
                               attrs={'long_name': 'time_bounds', 'standard_name': 'time_bounds'})
longitude = ds.longitude
latitude = ds.latitude

aatsr_data = xarray.open_mfdataset(aatsr_files, combine='nested', concat_dim='time')
aatsr_cld_phase_fraction = ice_liq_cld_fraction(aatsr_data)  # compute liquid and ice cloud fractions
aatsr_all_data = xarray.merge([aatsr_data, aatsr_cld_phase_fraction], compat='equals')
for modis_grp, aatsr_var in match_dir.items():
    out_file = extract_modis_dir / f"{modis_grp}.nc"
    out_aatsr_file = extract_aatsr_dir / f"{modis_grp}.nc"
    if out_file.exists() and out_aatsr_file.exists() and not args.overwrite:
        my_logger.warning(
            f'Skipping {out_file} and {out_aatsr_file} as both already exists and overwrite is not enabled.')
        continue
    with xarray.open_mfdataset(modis_files, combine='nested', concat_dim='time', group='/' + modis_grp) as modis_read:
        # reorder dims to be sensible and then assign coordinates
        modis = modis_read.transpose('time', 'latitude', 'longitude').assign_coords(time=time, longitude=longitude,
                                                                                    latitude=latitude)
        modis['time_bounds'] = time_bounds
        modis = modis[['Mean', 'time_bounds']].rename(Mean=modis_grp)  # Rename the variable to match AATSR variable
        modis = modis.compute()  # Compute the dataset to ensure all data is loaded

        # add in metadata
        modis = modis.assign_attrs(modis_attrs)  # Assign the attributes from the first file

        modis.to_netcdf(out_file, format='NETCDF4')  # Save the MODIS data to a netCDF file
        my_logger.info(f'Processed MODIS data written to {out_file}')
        # check for overlap.
        overlap = np.intersect1d(modis.time.values, aatsr_all_data.time.values)
        has_overlap = overlap.size > 0
        if has_overlap:  # got some overlapping data.
            my_logger.info('Overlapping data found between MODIS and AATSR datasets. Regridding and Masking')

            aatsr = aatsr_all_data[aatsr_var].rename(lon='longitude', lat='latitude').sel(
                time=slice(modis.time.min(), modis.time.max()))  # Load the AATSR variable
            aatsr = aatsr.rename(modis_grp)  # Rename to match MODIS variable
            aatsr = aatsr.regrid.conservative(modis[modis_grp]).compute()  # regrid to MODIS grid.
            aatsr = aatsr.where(~np.isnan(modis[modis_grp]), drop=True)  # mask aatsr data by MODIS data
            time_bounds_aatsr = modis['time_bounds'].sel(time=aatsr.time)  # Get the time bounds for AATSR

            history = aatsr_data.attrs.get('history',
                                           '') + f' Re-gridded and masked AATSR data for {modis_grp} using {pathlib.Path(__file__).name}.'
            attrs = aatsr_data.attrs.copy()
            attrs.update(history=history)
            aatsr = xarray.Dataset({
                modis_grp: aatsr,
                'time_bounds': time_bounds_aatsr}).assign_attrs(
                **attrs)  # Create a new dataset with the regridded AATSR data and time bounds

            aatsr.to_netcdf(out_aatsr_file, format='NETCDF4')
            my_logger.info(f'Processed AATSR data written to {out_aatsr_file}')
