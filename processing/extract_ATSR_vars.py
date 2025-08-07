#!//usr/bin/env python
# Convert fraction total cloud cover to liquid and ice cloud fractions and write to a new file.
import pathlib
import xarray
import netCDF4
import logging
import types

# Set up logging
my_logger = logging.getLogger(__name__)
if my_logger.hasHandlers():
    my_logger.handlers.clear()
my_logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

my_logger.addHandler(handler)
my_logger.propagate = False  # Prevents double logging if root logger is also configured

def ice_liq_cld_fraction(ds:xarray.Dataset) -> xarray.Dataset:
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
    liq_cld_frac = (cld_frac * liq_frac_total).assign_attrs(long_name='liquid cloud fraction',standard_name='liquid_water_cloud_area_fraction',units='1',valid_min=0.0, valid_max=1.0)
    ice_cld_frac = (cld_frac * (1-liq_frac_total)).assign_attrs(long_name='ice cloud fraction',standard_name='ice_cloud_area_fraction',units='1',valid_min=0.0, valid_max=1.0)
    # set the attributes for the new variables

    result = xarray.Dataset({
        'liquid_cloud_fraction': liq_cld_frac,
        'ice_cloud_fraction': ice_cld_frac,

    })
    result = result.assign_attrs(ds.attrs)
    result.attrs.update(history=[f'Extracted liquid and ice cloud fractions from AATSR cloud fraction data using {script_name}.']+[result.attrs.get('history','')])

    return result

file = pathlib.Path(r"data/200208-ESACCI-L3C_CLOUD-CLD_PRODUCTS-AATSR_ENVISAT-fv3.0.nc")
with netCDF4.Dataset(file, 'r') as nc: # get the format.
    file_format = nc.file_format
    my_logger.info(f'File format of {file} is {file_format}')


with xarray.open_dataset(file) as ds: # get in the date we want and process it.
    ice_l_cld_fraction = ice_liq_cld_fraction(ds).load()
output_file = file.parent / f"{file.stem}_il_cfract.nc"
ice_l_cld_fraction.to_netcdf(output_file,  format=file_format)
my_logger.info(f"Processed data written to {output_file}")


