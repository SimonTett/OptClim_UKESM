## plot the jacobian. This is done from the final config file.
import pathlib



import StudyConfig
import dfols
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

cfg_path = pathlib.Path("results/dfols26p/UKESM1_1_dfols26p_archer2_final.json")
cfg_path = pathlib.Path("opt_dfols46/UKESM1_1_dfols46p_archer2_final.json")
#cfg_path = pathlib.Path("opt_dfols26/UKESM1_1_dfols26p_archer2_final.json")
cfg = StudyConfig.readConfig(cfg_path)
soln = cfg.dfols_solution()
ranges = cfg.paramRanges().loc['rangeParam',:]
jac_trans = cfg.transJacobian()*ranges
err = cfg.Covariances(scale=True)['CovTotal']
std_err = pd.Series(np.sqrt(np.diag(err)),index=err.index)

transform = cfg.transMatrix(scale=True,dataFrame=True)
jac = jac_trans.T@transform # undo the transform

jac = jac.div(std_err,axis=1)
fig,axs = plt.subplots(nrows=1,ncols=1,num='jacobian',layout='constrained',clear=True,figsize=[12,8])
sns.heatmap(np.abs(jac),cmap='YlOrRd',ax=axs,vmin=2,vmax=500)
fig.show()
