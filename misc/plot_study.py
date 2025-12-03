#!/usr/bin/env python
# plot the progres of the study to date



import  model_base
import matplotlib.pyplot as plt
import pathlib
import argparse
import seaborn as sns
from matplotlib.colors import AsinhNorm
import matplotlib
import copy
from matplotlib.ticker import ScalarFormatter,LogFormatter,FixedLocator, FuncFormatter
import numpy as np

parser = argparse.ArgumentParser(description='Plot the study progress')
parser.add_argument('study_path', type=pathlib.Path, help='Path to the study config.')
parser.add_argument('--output', help='Output file for the plot', type=pathlib.Path, default=pathlib.Path('study_progress.jpg'))
args = parser.parse_args()

study = model_base.model_base.load(args.study_path)
obs = study.logical_obs(normalize=True)
cost = study.logical_cost()
params = study.logical_params(normalize=True)

## plotting
cmap = copy.copy(matplotlib.colormaps.get_cmap('RdYlGn'))
cmap.set_under('skyblue')
cmap.set_over('black')
sf = ScalarFormatter(useMathText=False)
sf.set_scientific(False)
fig, (ax_cost,ax_params,ax_obs) = plt.subplots(nrows=3,ncols=1,figsize=(6, 10),
                                               num='Study Progress', clear=True,sharex=True,layout='tight')
cmP=sns.heatmap(params.T, ax=ax_params,vmin=0,vmax=1,cbar=False,cmap=cmap,xticklabels=True)
ax_params.set_title("Normalised Parameters", fontsize='small')

cmO=sns.heatmap(obs.T, ax=ax_obs,norm=AsinhNorm(vmin=-20, vmax=20),cbar=False,cmap=cmap,xticklabels=True)
ax_obs.set_title("Normalised Obs misfit", fontsize='small')
cost.plot(ax=ax_cost,marker='o')

minv = cost.min()
minp = cost.argmin()  # use location in array (as that is what we plot)
ax_cost.set_title("Cost", fontsize='small')
a = ax_cost.plot(minp, minv, marker='o', ms=12, alpha=0.5)
ax_cost.axhline(minv, linestyle='dotted')
ax_cost.set_yscale('log')
ax_cost.set_xlim(-0.5,None)
#yt = np.round(ax_cost.get_yticks(),1)
#ax_cost.yaxis.set_major_locator(FixedLocator(yt))
nticks =int(np.ceil(np.log2(cost.max())))
upper= 2**nticks
lower = 2**int(np.floor(np.log2(cost.min())))
ax_cost.set_ylim(lower, upper)
yticks = np.geomspace(lower, upper, 6)  # adjust step as needed
ax_cost.yaxis.set_major_locator(FixedLocator(yticks))

# base LogFormatter (will produce readable labels for powers-of-two)
logf = LogFormatter(base=np.sqrt(2), labelOnlyBase=False)
# wrapper: prefer plain numeric labels for the actual tick values, else fall back to LogFormatter
fmt_fn = lambda x, pos: f"{x:.0f}" if ((x >= 1 and float(x).is_integer()) or x >= 10) \
    else ( logf(x, pos) if logf(x,pos) is not None else "")

fmt_fn = lambda x, pos: f"{x:.0f}" if (x >= 1 and float(x).is_integer()) or x >= 10 \
    else (logf(x, pos) if logf(x, pos) is not None else "")
def fmt_fn(x, pos):
    if (x >= 1 and float(x).is_integer()) or x >= 10:
        label = f"{x:.0f}"
    else:
        try:
            label = logf(x, pos)
        except AttributeError:
            label = ""

    return label

fn_form = FuncFormatter(fmt_fn)
ax_cost.yaxis.set_major_formatter(fn_form)
ax_cost.yaxis.set_minor_formatter(fn_form)
ax_cost.yaxis.minorticks_off()


ax_cost.yaxis.get_offset_text().set_visible(False)


for ax in (ax_params,ax_obs):
    ax.tick_params(axis='y', labelrotation=30, labelsize='x-small')
ax_obs.tick_params(axis='x', labelrotation=90, labelsize='x-small')
for cmm, title in zip([cmO, cmP], ['Obs', 'Param']):
    cb = fig.colorbar(cmm.collections[0], ax=ax_cost, label=title,orientation='horizontal', fraction=0.05, extend='both')
    cb.ax.xaxis.set_major_formatter(sf)
    cb.ax.xaxis.get_offset_text().set_visible(False)

fig.tight_layout()
fig.show()
fig.savefig(args.output,bbox_inches='tight')

## now to plot bonus plot of the individual obs contributions for the best run.
fig,ax = plt.subplots(nrows=1,ncols=1,num='Best run obs contributions', clear=True, figsize=(8, 6), layout='constrained')
best_loc = cost.argmin()
start = obs.iloc[0]
best = obs.iloc[best_loc]
start.plot.bar(ax=ax,color='black',label='Start',position=1,width=0.4)
best.plot.bar(ax=ax,color='red',label='Best',position=0,width=0.4)
for v in [-2,0,2]:
    ax.axhline(v,color='black',linestyle='dotted')
ax.tick_params(axis='x', labelrotation=60, labelsize='xx-small')
ax.set_title('Normalised Obs misfit for best & start runs')
ax.legend()
ax.set_yscale('asinh')


# after creating each colorbar (the variable `cb` in your loop)

ax.yaxis.set_major_formatter(sf)
ax.yaxis.get_offset_text().set_visible(False)
fig.show()








