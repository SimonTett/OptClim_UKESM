# plot simulated and observed calipso cloud fraction data for 2011.
import pathlib
import typing

import iris
import xarray
import matplotlib.pyplot as plt
import UKESMlib


def comp_calipso(files:list[pathlib.Path], region:typing.Optional[dict[str,slice]] = None) -> xarray.DataArray:
    """Compute the calipso cloud fraction from the model data files."""
    cubes = iris.load_cubes(files,['m01s02i371','m01s02i325'])
    msk=xarray.DataArray.from_iris(cubes[1])
    cf_msk=xarray.DataArray.from_iris(cubes[0])
    cf = cf_msk.where(msk > 1e-5)/msk


    if region is not None:
        cf = cf.sel(**region)
    wt = UKESMlib.compute_area_weights(cf)
    cf = cf.weighted(wt).mean(['longitude', 'latitude', 'time']).load()
    return cf

def comp_mn_temp(files:list[pathlib.Path],time_range:typing.Optional[slice]=None) -> float:
    """
    Compute the mean temperature from the model data files.
    :param time_range: time range to select
    :param files: list of files to process
    :return: avg temperature
    """
    temperature = iris.load_cube(files,'m01s03i236')
    temperature = xarray.DataArray.from_iris(temperature)
    wt = UKESMlib.compute_area_weights(temperature)
    temperature = temperature.weighted(wt).mean(['longitude','latitude']).load()
    if time_range is not None:
        temperature = temperature.sel(time=time_range)
    temperature = float(temperature.mean('time'))
    return temperature


region = dict(latitude=slice(-20,20),time=slice('2011-01-01','2011-12-30'))
files = sorted((UKESMlib.process_dir/"Example_model_data/tc000_140277").rglob('*.pz2011*.pp'))
cf_std = comp_calipso(files,region=region)
temp_std = comp_mn_temp(files,time_range=region['time'])

files = sorted((UKESMlib.process_dir/"Example_model_data/tc002_140277").rglob('*.pz2011*.pp'))
cf_perturb = comp_calipso(files,region=region)
temp_perturb = comp_mn_temp(files,time_range=region['time'])
# +4K runs.
files = sorted((UKESMlib.process_dir/"Example_model_data/tc004_224013").rglob('*.pz2011*.pp'))
cf_std_4K = comp_calipso(files,region=region)
temp_std_4K = comp_mn_temp(files,time_range=region['time'])
files = sorted((UKESMlib.process_dir/"Example_model_data/tc005_224013").rglob('*.pz2011*.pp'))
cf_perturb_4K = comp_calipso(files,region=region)
temp_perturb_4K = comp_mn_temp(files,time_range=region['time'])


## get the observed data
obs_files = sorted((UKESMlib.process_dir/"obs_data/CALIPSO_cld_area").glob('3D_CloudFraction330m_[0-9]*avg_CFMIP1_sat_3.1.2.nc'))

obs= xarray.open_mfdataset(obs_files)
ht = 1e3*obs.alt_mid.isel(time=0).squeeze(drop=True).drop_vars('time')
obs_calipso = obs.clcalipso.assign_coords(height=ht)


obs_tropical_mn = obs_calipso.sel(**region)
wt = UKESMlib.compute_area_weights(obs_tropical_mn)
obs_tropical_mn = obs_tropical_mn.weighted(wt).mean(['longitude','latitude','time']).load()






## now make plot
with plt.rc_context({'lines.linewidth': 2}):
    fig,(ax_values,ax_change)= plt.subplots(num='calipso',clear=True,ncols=2,figsize=(11.7,8.3),layout='constrained')
    cf_std.plot(label='Std UKESM1.1',y='height',ax=ax_values)
    cf_perturb.plot(label='Perturb UKESM1.1',y='height',linestyle='dashed',ax=ax_values)
    obs_tropical_mn.plot(label='Obs',y='height',ax=ax_values)

    ax_values.set_title('Tropical CALIPSO cloud fraction')
    ax_values.set_xlabel('Cloud Fraction')

    ## plot the +4K change
    delta_std = (cf_std_4K - cf_std)/(temp_std_4K-temp_std)
    delta_std.plot(label='+4K Std UKESM1.1',y='height',ax=ax_change)
    delta_perturb = (cf_perturb_4K - cf_perturb)/(temp_perturb_4K-temp_perturb)
    delta_perturb.plot(label='+4K Perturb UKESM1.1',y='height',linestyle='dashed',ax=ax_change)
    ax_change.set_title('Tropical CALIPSO cloud fraction change/K +4K')
    ax_change.axvline(0,color='black',linestyle='dotted')
    ax_change.set_xlabel('Cloud Fraction/K')
    for ax in (ax_values,ax_change):

        ax.set_ylabel('Height (m)')
        ax.legend()


    fig.show()
    fig.savefig('figures/calipso.png')

