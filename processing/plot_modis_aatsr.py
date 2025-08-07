# plot the MODIS/AATSR data
import  matplotlib.pyplot as plt
import xarray
import pathlib
import logging

my_logger = logging.getLogger(__name__)
if my_logger.hasHandlers():
    my_logger.handlers.clear()
my_logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

my_logger.addHandler(handler)
my_logger.propagate = False  # Prevents double logging if root logger is also configured
base_dir = pathlib.Path(r"P:\optclim_data\obs_data")
modis_dir = base_dir / "modis_cloud"
aatsr_dir = base_dir / "aatsr_cloud"
extract_modis_dir = base_dir / "modis_cloud_extract"
extract_aatsr_dir = base_dir / "aatsr_cloud_extract"
MODIS_files =list(extract_modis_dir.glob('*.nc'))


modis_ts = dict()
aatsr_ts = dict()
for file in MODIS_files:
    with xarray.open_dataset(file) as ds:
        name = file.stem
        modis_ts[name] = ds[name].mean(['latitude', 'longitude'])  # Get the mean over latitude and longitude
        my_logger.info(f'Processing MODIS file:{ file.name}')
    aatsr_file = extract_aatsr_dir / file.name # Assuming AATSR files have the same name as MODIS files
    with xarray.open_dataset(aatsr_file) as ds:
        name = aatsr_file.stem
        aatsr_ts[name] = ds[name].mean(['latitude', 'longitude'])
        my_logger.info(f'Processing AATSR  file:{ file.name}')
## now got the time series for MODIS and AATSR, plot them.
fig_mean, axs = plt.subplots(nrows=3, ncols=3, figsize=(11.7,8.3),sharex=True, clear=True,constrained_layout=True,num='MODIS_vs_AAATSR') # have 6 vars.
for ax, (name, modis_data) in zip(axs.flatten(),modis_ts.items()):
    m = modis_data.resample(time='YS-JAN')
    md = m.mean().where(m.count() == 12)  # Resample to annual mean, ignoring NaNs
    md.plot(ax=ax,  label='MODIS', color='blue')
    aatsr_data = aatsr_ts[name]
    a = aatsr_data.resample(time='YS-JAN')
    ad = a.mean().where(a.count() == 12)  # Resample to annual mean, ignoring NaNs
    ad.plot(ax=ax, label='AATSR', color='green')
    # add on a duplicate axis for the right hand side.
    ax2 = ax.twinx()  # Create a twin Axes sharing the x-axis
    (100*(md/ad)-100).plot(ax=ax2, label='AATSR', color='green', linestyle='--')  # Plot delta on the right y-axis
    ax2.set_ylabel('% Difference (MODIS - AATSR)')
    ax2.tick_params(axis='y', labelcolor='green')  # Set the color of the y-axis ticks
    ax2.axhline(0, color='green', linestyle='--', linewidth=0.5)  # Add a horizontal line at y=0
    ax2.label_outer()
    ax.set_title(name)
    ax.set_xlabel('Time')
    ax.set_ylabel(name)
    ax.tick_params(axis='x', rotation=45,labelsize='small')

fig_mean.show()
fig_mean.savefig('modis_vs_aatsr_mean.png', dpi=300, bbox_inches='tight')
## plot ice cloud fraction for MODIS and AATSR for July 2007
name = 'Cloud_Retrieval_Fraction_Ice'
modis_ice = xarray.open_dataset(extract_modis_dir/(name+'.nc'))[name].sel(time='2007-07')
modis_ice_zm = modis_ice.mean('longitude')
aatsr_ice = xarray.open_dataset(extract_aatsr_dir/(name+'.nc'))[name].sel(time='2007-07')
aatsr_ice_zm = aatsr_ice.mean('longitude')
kw_colourbar= dict(orientation='horizontal', pad=0.05, aspect=50, shrink=0.8, label='Ice Cloud Fraction')
fig_ice, (ax_map,ax) = plt.subplots(figsize=(11.7, 8.3), ncols=2,num='MODIS_vs_AAATSR_Ice_Cloud_Fraction',
                                    clear=True, constrained_layout=True)
# plot map.
(modis_ice-aatsr_ice).T.plot(ax=ax_map, cmap='viridis',robust=True,cbar_kwargs=kw_colourbar,)
ax_map.set_title('MODIS - AATSR Ice Cloud Fraction for July 2007')



modis_ice_zm.plot(ax=ax, label='MODIS Ice Cloud Fraction', color='blue',y='latitude')
aatsr_ice_zm.plot(ax=ax, label='AATSR Ice Cloud Fraction', color='green',y='latitude')
ax.set_title('Ice Cloud Fraction for July 2007')
ax.set_ylabel('Latitude')
ax.set_xlabel('Ice Cloud Fraction')
ax2 = ax.twiny()  # Create a twin Axes sharing the x-axis
ratio = (modis_ice_zm/aatsr_ice_zm-1.)*100
ratio.plot(ax=ax2,label='% Difference (MODIS - AATSR)', color='green', linestyle='--',y='latitude')
ax2.set_title(None)
ax2.set_xlabel('% Difference MODIS - AATSR')
ax2.tick_params(axis='x', labelcolor='green')  # Set the color of the y-axis ticks
ax2.axvline(0, color='green', linestyle='--', linewidth=0.5)  # Add a horizontal line at y=0


ax.tick_params(axis='x', rotation=45, labelsize='small')
ax.legend()
fig_ice.show()
fig_ice.savefig('modis_vs_aatsr_ice_cloud_fraction.png', dpi=300, bbox_inches='tight')