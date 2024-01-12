Software to support UKESM1.1 data processing
You need:
1) post_process/comp_sim_obs_UKESM1_1.py (wrapped by simobs_wrapper_UKESM1_1.sh)
   It expects to either run in the directory with PP files or 
   to be given the directory. See its documentation. 
2) configs/UKESM_1_1_opt.json -- config file
3) Set up the environment var OPT_ST_UKESM to point to the root dir for the UKESM stuff! See setup or setup_archer2

To test put your self in OPT_ST_UKESM and do:

./post_process/comp_sim_obs_UKESM1_1.py configs/configs/UKESM_1_1_opt.json  -d <PATH_TO_PP_DATA)

You can check the values in observations.json against the *standard* values in configs/UKESM_test_opt.json .

To produce the observed dataset proceed as follows -- on burn or stream  where
the observed data lives. 

 This dataset is actually the N48 values done for HadAM3 CMIP6
work. But for same period as post-processing will run for. Given we
are trialing approach this is reasonable.

1) Produce summary observed values. 
omp_obs_values.py configs/UKESM_1_1_opt.json output_obs2011.json ../HadCM3-CMIP6/data/Observations/*/*_N48*.nc --verbose

2) Re-process those to make tgt & covariance csv files.
make_obs_targets_covar.py output_obs2011.json tgt2011.json covariance/obserr2011.csv

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
