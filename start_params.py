
"""
Setup default model parameters
"""
# set env vars
import json
import pathlib
import UKESMlib
import collections
import datetime

out_file = pathlib.Path('configs/start_params_UKESM1_1.ijson')
wanted_namelists =['run_convection','run_cloud','run_precip','run_bl','jules_snow','run_radiation']
# list of namelists we want to use all variables from
extras = {
    "run_gwd":["gwd_frc","fbcd"],
    "run_ukca":["seadms_ems_scaling","sea_salt_ems_scaling","dry_depvel_so2_scaling","sigma_updraught_scaling"]
    }

pth = UKESMlib.expand('$OPTCLIMTOP/OptClimVn3/configurations/parameters_UKESM1_1.ijson')
with pathlib.Path(pth).open('rt') as f:
    default_params  = json.load(f)['defaultParams']

index_namelist = collections.defaultdict(list)
# work out for each namelist what parameters are there
for k in default_params.keys():
    if not  k.endswith('_namelist_comment'):
        continue # want the namelist_comment!
    variable = k.replace('_namelist_comment','') # get the variable name and check it exists
    if variable not in default_params:
        raise ValueError(f'No variable {variable} for {k}')
    namelist = default_params[k]
    index_namelist[namelist].append(variable)


vars_to_set=list() # list of variables to set
for name in wanted_namelists:
    for var in index_namelist[name]:
        vars_to_set.append(var)
## add on the extra variables.
for name,extra in extras.items():
    vars_to_set += extra
# now can actually work the variables and comments to write out
now = datetime.datetime.now(datetime.UTC)
start_values=dict(
    vars_comment=f'List of {len(vars_to_set)} variables set to default value. Set by {__file__} at {now}',
)
for var in vars_to_set:
    if var not in default_params:
        raise ValueError(f'No variable {var} in default params')
    start_values[var] = None # will use default value
    comment_key = var + '_comment'
    if comment_key in default_params:
        start_values[comment_key] = default_params[comment_key]
    else:
        start_values[comment_key] = f'No comment for {var}'

# now write out the parameters to a file
with out_file.open('wt') as f:
    json.dump(start_values,f,indent=2)

