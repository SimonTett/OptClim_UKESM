## plot the jacobian. This is done from the final config file.
import pathlib



import StudyConfig
import dfols
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm

import math

def ceil_preferred(x: float, preferred=(1, 2, 5)) -> float:
    if x <= 0:
        raise ValueError("x must be > 0")
    if not preferred:
        raise ValueError("preferred must not be empty")

    p = math.floor(math.log10(x))
    scale = 10 ** p
    m = x / scale

    vals = sorted(preferred)
    # include decade rollover
    k = next((v for v in vals if m <= v), vals[0] * 10)
    return k * scale

cfg_path = pathlib.Path("results/dfols26p/UKESM1_1_dfols26p_archer2_final.json")
cfg_path = pathlib.Path("opt_dfols46/UKESM1_1_dfols46p_archer2_final.json")
cfg_path = pathlib.Path("opt_dfols46_try2/UKESM1_1_dfols46p_archer2_try2_final.json")
#cfg_path = pathlib.Path("opt_dfols26/UKESM1_1_dfols26p_archer2_final.json")
cfg_path = pathlib.Path("results/dfols46p_try2/UKESM1_1_dfols46p_archer2_try2_final.json")
cfg_path = pathlib.Path("dfols46p_3/UKESM1_1_dfols46p_archer2_try3_final.json")
cfg_path = pathlib.Path("/work/n02/n02/egavilan/calibration/dfol_nemo_12p/NEMO36_dfols12p_archer2_final.json")
cfg = StudyConfig.readConfig(cfg_path)
soln = cfg.dfols_solution()
ranges = cfg.paramRanges().loc['rangeParam',:]
jac_trans = cfg.transJacobian()
err = cfg.Covariances(scale=True)['CovTotal']
std_err = pd.Series(np.sqrt(np.diag(err)),index=err.index)

transform = cfg.transform_matrix(scale=True,inverse=True)
jac = transform@jac_trans # undo the transform
jac = jac.mul(ranges,axis=1) # scale by the ranges.
jac = jac.div(std_err,axis=0) # how many std errors does a  unit normalised param change do.
vmax = ceil_preferred(float(np.abs(jac).max(axis=None,numeric_only=True)))
fig,axs = plt.subplots(nrows=1,ncols=1,num='jacobian',layout='constrained',clear=True,figsize=[12,8])

sns.heatmap(jac,cmap='Spectral',ax=axs,norm=AsinhNorm(vmin=-vmax,vmax=vmax))
axs.tick_params(axis='x', labelrotation=45) 
fig.show()
