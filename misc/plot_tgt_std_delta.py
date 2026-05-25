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
import StudyConfig
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

config = StudyConfig.readConfig('$OPT_UKESM_ROOT/configs/UKESM1_1_dfols46p_archer2_try3.json') # config


std = config.standardObs(scale=True)
tgt  = config.targets(scale=True).drop('Cess') # drop Cess as not in amip till run +4K


cov = config.Covariances(scale=True)

#obsErr = pd.read_csv(obsErr_path,index_col=0).reindex(index=std.index,columns=std.index)
obsErr = cov['CovObsErr']
intErr  = cov['CovIntVar']
totalErr = cov['CovTotal']
std_err = pd.Series(np.sqrt(np.diag(totalErr)),index=totalErr.index)
obs_err = pd.Series(np.sqrt(np.diag(obsErr)),index=obsErr.index)
delta = (std - tgt)

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

fig_std,axs_std = plt.subplots(nrows=nx, ncols=ny,num=f'Std Model Bias {year}',
                      clear=True,figsize=fsize,layout='constrained')
for ax, ax_tgt,ax_std,gdelta in zip(axs.flatten(),axs_tgt.flatten(),axs_std.flatten(),grouped):
    new_indx = gdelta.index.str.replace(r'^[A-Za-z0-9@]+_([0-9]+_)?','',regex=True)
    rename_index = dict(zip(gdelta.index,new_indx))
    serr = std_err.reindex(gdelta.index).rename(rename_index)
    oerr = obs_err.reindex(gdelta.index).rename(rename_index)
    dd= gdelta.rename(rename_index)
    dd_std = dd/serr # how many std errors.
    dtgt = tgt.reindex(gdelta.index).rename(rename_index)
    dsim = std.reindex(gdelta.index).rename(rename_index)

    dd.plot.bar(ax=ax,yerr=serr,capsize=3,position=0.5)
    dd_std.plot.bar(ax=ax_std,position=0.5) # plot the number of std errors.
    dtgt.plot.bar(ax=ax_tgt,position=0.0,width=0.2,color='green')
    dsim.plot.bar(ax=ax_tgt,position=0.6,color='red',width=0.2)
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
    for a in [ax,ax_tgt,ax_std]:
        a.tick_params(axis='x', labelrotation=0, labelsize='small')
        a.set_title(gdelta.name)
        a.axhline(0,color='black',linestyle='--')
for f,file in zip([fig,fig_tgt,fig_std],[f'model_bias_{year}.png',f'obs_tgt_{year}.png',f'std_model_bias_{year}.png']):
    f.tight_layout()
    f.show()
    f.savefig(figures_dir/file)
