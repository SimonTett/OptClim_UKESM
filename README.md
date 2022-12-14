Software to support UKESM1 data processing
You need:
1) comp_sim_obs_UKESM_atmos.py which actually does the data post processing. 
   It expects to either run in the directory with netcdf files or 
   to be given the directory. See its documentation. 
2) UKESM_test_opt.json -- config file
3) Set up the environment var OPT_UKESM to point to the root dir for the UKESM stuff!

To test put your self in OPT_UKESM and do:

./post_process/comp_sim_obs_UKESM_atmos.py configs/UKESM_test_opt.json -d test_data/test_UKESM_nc/History_Data/

You can check the values in observations.json against the *standard* values in configs/UKESM_test_opt.json .

To produce the observed dataset proceed as follows -- on burn or stream  where
the observed data lives. 

 This dataset is actually the N48 values done for HadAM3 CMIP6
work. But for same period as post-processing will run for. Given we
are trialing approach this is reasonable.

1) Produce summary observed values. 
comp_obs_values.py configs/UKESM_test_opt.json output_obs2012.json  ../HadCM3-CMIP6/data/Observations/*/*_N48*.nc --verbose

2) Re-process those to make tgt & covariance csv files.
./make_obs_targets_covar.py output_obs2012.json tgt2012.json covariance/obserr2012.csv

3) Include the tgt in the main config file. 
And remember to commit/push to git.

Directories:

configs -- all things configuration related

covariance -- where covariances live

post_process -- all things post processing related

test_data -- where test data lives. 

rose_configs -- rose configurations

baseSuite - the suites used for the example study.
- Lbase254: base suite for the model (after adding the pre and post model tasks): this is cloned for each run.
- Lctl-254: the controlling suite that causes the clones to be created.

results  -- the final json (...final.json) and the plot summarising the results
