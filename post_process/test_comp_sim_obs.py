# Code to test comp_sim_obs_UKEMS1_1 works.

import logging
from  comp_sim_obs_UKESM1_1 import *
import tempfile
logging.basicConfig(level='INFO')
rootdir=pathlib.Path(r"C:\Users\stett2\OneDrive - University of Edinburgh\data\Opt_UKESM1.1")
files = list(rootdir.glob('*a.pm*.pp'))
with tempfile.NamedTemporaryFile(suffix='.json',delete=False) as tfile: # whne upgrade to 3.12 delete -> delete_on_close
    tfile.close()
    results = compute_values(files,pathlib.Path(tfile.name),land_mask_fraction=0.5)

#
for name, value in results.items():
    print(f"{name}: {value:.4g}")
print("============================================================")



