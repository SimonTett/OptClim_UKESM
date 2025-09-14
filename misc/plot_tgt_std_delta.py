# plot the difference between the AMIP data and the obs data for specified year
year='2011'
#year='2005'
figures_dir = 'figures'
import UKESMlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import typing
import pathlib
figures_dir = pathlib.Path(figures_dir)
figures_dir.mkdir(parents=True, exist_ok=True)
def comp_x_y(n:int,max_delta:typing.Optional[int]=None) -> tuple[int,int]:
    """Compute a good layout for n subplots."""
    if n <= 1:
        return 1,1
    # try to get as square as possible
    x = math.floor(math.sqrt(n))
    y= n//x
    if max_delta is None:
        max_delta = x//2
    while (y*x)<n:
        if abs(y-x)>max_delta:
            y += 1
        else:
            x +=1
    return x,y

AMIP_pth =UKESMlib.expand(f'$OPT_UKESM_ROOT/configs/std_AMIP_{year}.ijson')
obs_pth = UKESMlib.expand(f'$OPT_UKESM_ROOT/configs/target_{year}.ijson')
scaling_path = UKESMlib.expand('$OPT_UKESM_ROOT/configs/scalings.ijson')
obsErr_path = UKESMlib.expand("$OPT_UKESM_ROOT/covariance/obserr.csv")
intVar_path = UKESMlib.expand("$OPT_UKESM_ROOT/covariance/AMIP_intvar.csv")

std = pd.read_json(AMIP_pth,typ='series')
tgt = pd.read_json(obs_pth,typ='series').drop('Cess') # drop Cess as not in amip till run +4K
scalings = pd.read_json(scaling_path,typ='series').reindex(std.index).fillna(1.0)
tgt = tgt.reindex(std.index)
# fix AOD_550 to have unit scaling
aod_indx = [k for k in scalings.index if k.startswith('AOD_550')]
scalings.loc[aod_indx] = 1.0

obsErr = pd.read_csv(obsErr_path,index_col=0).reindex(index=std.index,columns=std.index)
intErr = pd.read_csv(intVar_path,index_col=0).reindex(index=std.index,columns=std.index)
totalErr = (obsErr+2*intErr)
std_err = pd.Series(np.sqrt(np.diag(totalErr)),index=totalErr.index)*scalings
obs_err = pd.Series(np.sqrt(np.diag(obsErr)),index=obsErr.index)*scalings
delta = (std - tgt)*scalings
scale_tgt = tgt * scalings
scale_sim = std * scalings
frac_delta = 100.0*delta/delta-100.0
var_names = delta.index.str.extract(r'^([A-Za-z0-9@]+)_', expand=False)
grouped = [group.rename(name) for name, group in delta.groupby(var_names)]

## and plot
nx,ny = comp_x_y(len(grouped))
fsize=(11.8,9)
fig,axs = plt.subplots(nrows=nx, ncols=ny,num=f'Model Bias {year}',
                      clear=True,figsize=fsize,layout='constrained')
fig_tgt,axs_tgt = plt.subplots(nrows=nx, ncols=ny,num=f'Obs tgt {year}',
                      clear=True,figsize=fsize,layout='constrained')
for ax, axs_tgt,gdelta in zip(axs.flatten(),axs_tgt.flatten(),grouped):
    new_indx = gdelta.index.str.replace(r'^[A-Za-z0-9@]+_([0-9]+_)?','',regex=True)
    rename_index = dict(zip(gdelta.index,new_indx))
    serr = std_err.reindex(gdelta.index).rename(rename_index)
    oerr = obs_err.reindex(gdelta.index).rename(rename_index)
    dd= gdelta.rename(rename_index)
    dtgt = scale_tgt.reindex(gdelta.index).rename(rename_index)
    dsim = scale_sim.reindex(gdelta.index).rename(rename_index)
    dd.plot.bar(ax=ax,yerr=serr,capsize=3,position=0.5)
    dtgt.plot.bar(ax=axs_tgt,position=0.0,width=0.2,color='green')
    dsim.plot.bar(ax=axs_tgt,position=0.6,color='red',width=0.2)
    # add 2nd error bar
    x = np.arange(len(gdelta))
    ax.errorbar(
        x,
        dd.values,
        yerr=oerr.values,
        fmt='none',
        ecolor='red',
        capsize=5
    )
    for a in [ax,axs_tgt]:
        a.tick_params(axis='x', labelrotation=0, labelsize='small')
        a.set_title(gdelta.name)
        a.axhline(0,color='black',linestyle='--')
for f,file in zip([fig,fig_tgt],[f'model_bias_{year}.png',f'obs_tgt_{year}.png']):
    f.tight_layout()
    f.show()
    f.savefig(figures_dir/file)
