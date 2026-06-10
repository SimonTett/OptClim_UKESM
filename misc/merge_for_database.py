# Merge 26p, 46p & 46p_try2 cases to produce one large evaluation database.
import pandas as pd
from genericLib import expand
import json
import datetime

combined_obs = []
combined_params = []

# output files
out_param_file = "$OPT_UKESM_ROOT/eval_db/params_combined_46p_try3.csv" # params
out_obs_file = "$OPT_UKESM_ROOT/eval_db/obs_combined_46p_try3.csv" # obs
out_json_file = "$OPT_UKESM_ROOT/eval_db/combined_46p_try3.ijson" # config


# Input files
obs_files =  ["$OPT_UKESM_ROOT/eval_db/obs26p.csv","$OPT_UKESM_ROOT/eval_db/obs46p.csv","$OPT_UKESM_ROOT/eval_db/obs46p_try2.csv"] # input obs files to combine
param_files = ["$OPT_UKESM_ROOT/eval_db/params26_26p.csv","$OPT_UKESM_ROOT/eval_db/params46_46p.csv","$OPT_UKESM_ROOT/eval_db/params46p_try2.csv"] # input param files
for file,param_file in zip(obs_files,param_files):

    f= expand(file)
    
    obs =  pd.read_csv(f,index_col=[0])
    name = f.stem.replace('obs','')
    index = [name+'_'+idx for idx in obs.index]
    obs.index = index
    combined_obs.append(obs)
    
    f2=expand(param_file)
    params = pd.read_csv(f2,index_col=[0])
    pindex = [name+'_'+idx for idx in params.index]
    # verify obs and params have same index.
    if set(pindex) != set(index):
        raise ValueError(f"Params index not the same as obs index for {f} and {f2}")
    params.index=pindex
    combined_params.append(params)


combined_obs = pd.concat(combined_obs) # combine the obs datasets.
combined_params = pd.concat(combined_params) # combine the params datasets.
combined_params.to_csv(expand(out_param_file))
combined_obs.to_csv(expand(out_obs_file))
# generate the evaluation_db metadata.
eval_db = {
    "parameters": out_param_file,
    "simulated_observations": out_obs_file,
    "start_index": "min",
    "_comment": f"Generated using {__file__} at {datetime.datetime.now()} merging 26p, 46p & 46p_try2cases",
    "obs_files_comment":obs_files,
    "param_files_comment":param_files
  }

eval_config = {"evaluation_database":eval_db} # wrap it in evaluation_database
out_file=expand(out_json_file)
with open(out_file, 'wt') as f:
    json.dump(eval_config,f,indent=2)
print(f"Saved evaluation database to {out_file}")
    
