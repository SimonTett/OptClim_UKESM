# Little script to look at covariance matrix
# Plots diagonal values and transMatrix
import numpy as np
import  seaborn as sns
import StudyConfig
import genericLib
import matplotlib.pyplot as plt
import  matplotlib.colors
import logging


# code to get cov...

# 1) Bootstrap early so pre-config warnings are formatted consistently
logging.basicConfig(
    level=logging.WARNING,  #
    format="BS:%(levelname)s:%(name)s:%(message)s"
)
config_path = genericLib.expand("$OPT_UKESM_ROOT/configs/UKESM1_1_dfols46p_archer2_try3.json")

config = StudyConfig.readConfig(config_path,check=False)

log_config = config.logging_config(remove_file_handlers=True)
# drop all the file logging as we don't need it.

genericLib.setup_logging('INFO',log_config=log_config)
config.check()
tMat = config.transform_matrix(scale=True)
cov = config.Covariances(scale=True)['CovTotal']
## plot covariance matrix
plt.figure(figsize=(10, 8),clear=True,num='Covariance Matrix')
# get ranges and make them sym. Pick 90% of values a bit different from zero
acov = np.abs(cov)
bound = np.quantile(np.abs(np.diag(cov)),0.95)
#bound = np.max(np.abs(cov))
ticks = np.geomspace(bound/10,bound,5)
ticks = 2**np.ceil(np.log2(ticks))
ticks = np.concatenate([-ticks[::-1],[0],ticks])
bound = 2**np.ceil(np.log2(bound))
lin_width = np.ceil(bound/100)
# Copy cmap so you don't modify the global "coolwarm"
cmap = plt.get_cmap("coolwarm").copy()
cmap.set_under("black")  # color for values < vmin
cmap.set_over("orange")   # color for values > vmax
cmap.set_under("#313695")  # color for values < vmin
cmap.set_over("#a50026")   # color for values > vmax
cmap.set_over("#8B0000")   # darkred
cmap.set_under("#001A66")  # dark blue
sns.heatmap(
    cov,
    cmap=cmap,
    norm=matplotlib.colors.AsinhNorm(vmin=-bound, vmax=bound,linear_width=lin_width),
    #mask=mask,
    cbar_kws={'label': 'Covariance ','ticks': ticks, 'format': '%.2g',"extend":'both'},

    square=True
)
plt.title("Covariance Matrix (asinh Color Scale)")
plt.xlabel("Obs")
plt.ylabel("Obs")
plt.tick_params(axis='x', labelrotation=90, labelsize='small')
plt.tick_params(axis='y', labelrotation=0, labelsize='small')
plt.tight_layout()
plt.show()

## Plot the transform matrix.

trans_matrix = config.transform_matrix(scale=True)
plt.figure(figsize=(10, 8),clear=True,num='Transform Matrix')
# get ranges and make them sym. Pick 90% of values a bit different from zero
bound = np.quantile(np.abs(cov[np.abs(cov)>0.1]),0.99)
bound = np.max(np.abs(trans_matrix))

ticks = np.geomspace(bound/10,bound,5)
ticks = 2**np.ceil(np.log2(ticks))
ticks = np.concatenate([-ticks[::-1],[0],ticks])
bound = 2**np.ceil(np.log2(bound))
sns.heatmap(
    trans_matrix,
    cmap="coolwarm",
    norm=matplotlib.colors.AsinhNorm(vmin=-bound, vmax=bound,linear_width=2),
    #mask=mask,
    cbar_kws={'label': 'Transform ','ticks': ticks, 'format': '%.2g'},

    square=True
)
plt.title("Transform Matrix (asinh Color Scale)")
plt.ylabel("Index")
plt.xlabel("Obs")

plt.tick_params(axis='x', labelrotation=90, labelsize='small')
plt.tight_layout()

plt.show()





