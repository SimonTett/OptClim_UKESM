import iris
import UKESMlib
import numpy as np
import iris.quickplot 
output_netcdf = UKESMlib.expand('$OPT_UKESM_ROOT/data/sst_amip_p4_n96e.nc')
output_ice_netcdf = UKESMlib.expand('$OPT_UKESM_ROOT/data/sic_amip_n96e.nc')
output_pp = UKESMlib.expand('$OPT_UKESM_ROOT/data/sst_amip_p4_n96e.pp')
sst_file = '/work/y07/shared/umshared/CMIP6_ANCIL/data/ancils/n96e/timeseries_1870-2016/SstSeaIce/sst_amip_n96e.anc'
ice_file = '/work/y07/shared/umshared/CMIP6_ANCIL/data/ancils/n96e/timeseries_1870-2016/SstSeaIce/seaice_amip_n96e.anc'
sst = iris.load_cube(sst_file)
ice = iris.load_cube(ice_file)
sstp4 = sst+np.where(np.abs(ice.data) < 0.3, 4,0)
sstp4.rename('surface_temperature')
iris.save(sstp4,str(output_netcdf),fill_value=sstp4[-1].data.data.min())
iris.save(ice,str(output_ice_netcdf),fill_value=ice[-1].data.data.min())


