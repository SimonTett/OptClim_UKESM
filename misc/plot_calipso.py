# plot simulated and observed calipso cloud fraction data for 2011.
import iris
import xarray
import matplotlib.pyplot as plt
import UKESMlib
import glob

region = dict(latitude=slice(-20,20),time=slice('2011-01-01','2011-12-30'))


files=sorted(glob.glob('/gws/nopw/j04/terrafirma/tetts/um_archive/20*/*a.p5*.pp'))
cubes = iris.load_cubes(files,['m01s02i371','m01s02i325'])
msk=xarray.DataArray.from_iris(cubes[1])
cf_msk=xarray.DataArray.from_iris(cubes[0])
cf = cf_msk.where(msk > 1e-5)/msk
wt=UKESMlib.compute_area_weights(cf)
tropical_mn = cf.sel(**region)
tropical_mn = tropical_mn.weighted(wt).mean(['longitude','latitude','time']).load()

## get the Cobserved data
obs_files = sorted(glob.glob('/gws/nopw/j04/terrafirma/tetts/data/obs_data/CALIPSO_cld_area/3D_CloudFraction330m_[0-9]*avg_CFMIP1_sat_3.1.2.nc'))

obs= xarray.open_mfdataset(obs_files)
ht = 1e3*obs.alt_mid.isel(time=0).squeeze(drop=True).drop_vars('time')
obs_calipso = obs.clcalipso.assign_coords(height=ht)

wt = UKESMlib.compute_area_weights(obs_calipso)
obs_tropical_mn = obs_calipso.sel(**region)
obs_tropical_mn = obs_tropical_mn.weighted(wt).mean(['longitude','latitude','time']).load()






## now make plot
fig = plt.figure(num='calipso',clear=True)
tropical_mn.plot(label='UKESM1.1',y='height')
obs_tropical_mn.plot(label='Obs',y='height')
plt.title('CALIPSO cloud area')
plt.legend()
fig.show()

