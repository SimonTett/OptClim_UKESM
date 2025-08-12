#!/usr/bin/env python
## make a confy file for post-processing.
import json
import pathlib


pp_dir = pathlib.Path(__file__).parent
pp_file = pp_dir / 'pp_config_test.json'
script = pp_dir /'comp_sim_obs_UKESM1_1.py'
land_sea = pp_dir / 'land_frac.nc'
post_process = dict(
    script=str(script),
    output_file='observations.json',
    mask_file=str(land_sea),
    mask_file_comment="Path for landfrac file.",
    mask_fraction=0.5,
    mask_fraction_comment="Critical Fraction. Specify if mask  is a land/sea fraction. Values >= are land < sea. Set to null if mask is a t/f mask",
    start_time=None,
    start_time_comment="Start time as ISO std string. ",
    end_time="2011-12-31",
    end_time_comment="End time as str of ISO std string",
    file_pattern='*a.p[5m]*.pp',
    file_pattern_comment="File pattern to match for post-processing. Use * as wildcard. ",

)

with pp_file.open('wt') as f:
    json.dump(post_process, f, indent=2)

print(f"Post-processing config file created at {pp_file}")
