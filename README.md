Software to support UKESM1 data processing.


To produce the observed dataset proceed as follows -- on burn where
data lives. 

 This dataset is actually the N48 values done for HadAM3 CMIP6
work. But for same period as post-processing will run for. Given we
are trialing approach this is reasonable.

1) Produce summary observed values. 
comp_obs_values.py configs/UKESM_test_opt.json output_obs2012.json  ../HadCM3-CMIP6/data/Observations/*/*_N48*.nc --verbose

2) Re-process those to make tgt & covariance csv files.
./make_obs_targets_covar.py output_obs2012.json tgt2012.json covariance/obserr2012.csv

3) Include the tgt in the main config file. 
And remember to commit/push to git.

