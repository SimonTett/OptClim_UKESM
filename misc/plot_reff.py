# Plot Reff from UM using 'main' calculation and 'COSP' calculation
import iris
import matplotlib.pyplot as plt
import xarray
from collections import Counter, defaultdict
import warnings



import UKESMlib
import pathlib

## load data
base_dir = pathlib.Path(r"C:\Users\stett2\OneDrive - University of Edinburgh\data\Opt_UKESM1\Example_model_data\u-dr157\20050701T0000Z")
file_str = "2005jul"
files_pm= [base_dir/f"dr157a.pm{file_str}.pp"]
files_cosp = [base_dir/f"dr157a.p5{file_str}.nc"]
stash_codes_sim  = ['m01s01i245', 'm01s01i246'] # Sim Reff, SIm wt
stash_codes_cosp = ['m01s02i463', 'm01s02i330'] # COSP Reff, COSP wt
with warnings.catch_warnings():
    warnings.simplefilter("ignore",category=FutureWarning)
    cubes_pm = UKESMlib.um_cubes(files_pm, stash_codes=stash_codes_sim)
cubes_cosp = xarray.open_mfdataset(files_cosp)
## get in the  observations
Reff_modis = xarray.open_mfdataset(r"C:\Users\stett2\OneDrive - University of Edinburgh\data\Opt_UKESM1\obs_data\modis_cloud_extract\Cloud_Particle_Size_Liquid.nc").Cloud_Particle_Size_Liquid.sel(time='2005-07').load()
Reff_aatsr = xarray.open_mfdataset(r"C:\Users\stett2\OneDrive - University of Edinburgh\data\Opt_UKESM1\obs_data\aatsr_cloud_extract\Cloud_Particle_Size_Liquid.nc"). Cloud_Particle_Size_Liquid.sel(time='2005-07').load()
CTP_modis = xarray.open_mfdataset(r"C:\Users\stett2\OneDrive - University of Edinburgh\data\Opt_UKESM1\obs_data\modis_cloud_extract\Cloud_Top_Pressure.nc").Cloud_Top_Pressure.sel(time='2005-07').load()
CTP_aatsr = xarray.open_mfdataset(r"C:\Users\stett2\OneDrive - University of Edinburgh\data\Opt_UKESM1\obs_data\aatsr_cloud_extract\Cloud_Top_Pressure.nc").Cloud_Top_Pressure.sel(time='2005-07').load()
## index
stash_counts = Counter()
stash_names = defaultdict(str)
cubes_sim= dict()
for cube in cubes_pm:
    # Get STASH code from attributes (if present)
    stash_code = str(cube.attributes.get('STASH', 'Unknown'))
    cubes_sim[stash_code] = xarray.DataArray.from_iris(cube) # convert to a xarray object so can do things with it

## Compute Reff
Reff = cubes_sim['m01s01i245'].where(cubes_sim['m01s01i246']>1e-5)/cubes_sim['m01s01i246'] # Sim Reff
cosp_reff_wt = cubes_cosp['m01s02i452']
Reff_cosp = 1e6*cubes_cosp['m01s02i463'].where(cosp_reff_wt>0)/cosp_reff_wt #
cosp_ctp_wt = cubes_cosp['m01s02i451']
CTP_cosp = cubes_cosp['m01s02i465'].where(cosp_ctp_wt>0)/cosp_ctp_wt # COSP cloud top pressure

## and then plot them
fig,(ax,ax_Ctp) = plt.subplots(ncols=2,clear=True,num='Comparison',layout='constrained',figsize=(8,6))
for da,label,color in [(Reff_modis,'MODIS', 'black'),(Reff_aatsr,'AATSR','grey')]:
    da.squeeze(drop=True).mean('longitude').plot(ax=ax,label=label,color=color)
Reff.mean('longitude').plot(ax=ax,label='Std Reff',color='red')
Reff_cosp.mean('longitude').plot(ax=ax,label='COSP Reff',color='blue')
# plot the CTP
(0.01*CTP_cosp).mean('longitude').plot(ax=ax_Ctp,label='COSP CTP',color='blue')
for da,label,color in [(CTP_modis,'MODIS', 'black'),(CTP_aatsr,'AATSR','grey')]:
    da.squeeze(drop=True).mean('longitude').plot(ax=ax_Ctp,label=label,color=color)
ax.set_title('Reff Comparison')
ax.set_ylabel('Reff (microns)')
ax.legend()
ax_Ctp.set_ylabel('CT Pressure (hPa)')
ax_Ctp.set_title('COSP Cloud Top Pressure')
ax_Ctp.set_ylim(1000,100)
ax_Ctp.legend()
fig.show()
fig.savefig('Reff_CTP_comparison.png',dpi=300)

