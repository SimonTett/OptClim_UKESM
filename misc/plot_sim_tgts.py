# Plot simulated values from optimisation study, best value and tgt
from setuptools.command.rotate import rotate

import SubmitStudy
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import  typing
import math
import pandas as pd



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




start_idx = 'I0_i0' # initial params
cfg_path = list(pathlib.Path(r"C:\Users\geosfeld\OneDrive - University of Edinburgh\Documents\dfols26p").glob('*.scfg'))[0]
cfg = SubmitStudy.SubmitStudy.load_SubmitStudy(cfg_path)
best_idx = cfg.logical_cost().idxmin()

sim_obs = cfg.logical_obs(scale=True)
sim_norm_obs = cfg.logical_obs(scale=True,normalize=True)
start = sim_obs.loc[start_idx]
best = sim_obs.loc[best_idx]
start_norm = sim_norm_obs.loc[start_idx]
best_norm = sim_norm_obs.loc[best_idx]

tgt_obs = cfg.config.targets(scale=True)
cov = cfg.config.Covariances(scale=True)

indx = tgt_obs.index
def get_err(cov:pd.DataFrame,indx) -> pd.Series:
    err = cov.reindex(index=indx,columns=indx)
    err = np.sqrt(np.diag(err))
    err = pd.Series(err,index=indx)
    return err

obsErr = get_err(cov['CovObsErr'],indx)
intErr = get_err(cov['CovIntVar'],indx)
totalErr = get_err(cov['CovTotal'],indx)

delta_best = best - tgt_obs
delta_start = start - tgt_obs

var_names = tgt_obs.index.str.extract(r'^([A-Za-z0-9@]+)_', expand=False).fillna('Cess')
# add in Cess
grouped = [group.rename(name) for name, group in tgt_obs.groupby(var_names)]

## and plot
year=2011
nx,ny = comp_x_y(len(grouped))
fsize=(11.8,9)
fig,axs = plt.subplots(nrows=nx, ncols=ny,num=f'Sim tgt {year}',
                      clear=True,figsize=fsize,layout='constrained')
fig_tgt,axs_norm = plt.subplots(nrows=nx, ncols=ny,num=f'Norm Sim - Tgt {year}',
                      clear=True,figsize=fsize,layout='constrained')
legend = True
for ax_tgt,ax_norm,gdelta in zip(axs.flatten(),axs_norm.flatten(),grouped):
    new_indx = gdelta.index.str.replace(r'^[A-Za-z0-9@]+_([0-9]+_)?','',regex=True)
    rename_index = dict(zip(gdelta.index,new_indx))
    serr = totalErr.reindex(gdelta.index).rename(rename_index)
    interr = intErr.reindex(gdelta.index).rename(rename_index)
    nbest= best_norm.reindex(gdelta.index).rename(rename_index).rename('Best')
    nstart = start_norm.reindex(gdelta.index).rename(rename_index).rename('Start')
    tgt = tgt_obs.reindex(gdelta.index).rename(rename_index).rename('Target')
    sim = start.reindex(gdelta.index).rename(rename_index).rename('Start')
    bst = best.reindex(gdelta.index).rename(rename_index).rename('Best')
    nerr = math.sqrt(2)*interr / serr



    wdth=0.8
    plot_df = pd.DataFrame([sim,bst,tgt])

    plot_df.T.plot.bar(ax=ax_tgt,color=['red','cornflowerblue','green'],
                       width=wdth,legend=legend,yerr=serr.to_frame('Target'),capsize=3,ecolor='k')
    norm_df = pd.DataFrame([nstart,nbest])
    norm_df.T.plot.bar(ax=ax_norm,color=['red','cornflowerblue'],width=wdth,legend=legend,yerr=nerr.to_frame('Best'),capsize=3,ecolor='k')
    legend=False

    for a in [ax_norm,ax_tgt]:
        a.tick_params(axis='x', labelrotation=0, labelsize='small')
        a.set_title(gdelta.name)
        a.axhline(0,color='black',linestyle='--')
        nx = len(nbest)
        a.set_xlim(-1,nx)
        a.tick_params(axis='x', labelrotation=45)
    for v in [-2,2]:
        ax_norm.axhline(v,color='black',linestyle=':')


fig_tgt.show()
fig.show()

