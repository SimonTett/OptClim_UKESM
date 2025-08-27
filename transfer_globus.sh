# script to transfer to jasminusing globus.
# note uses earlier python. Paths need (I think) 
archer_dir=$1 ; shift
jasmin_dir=$1 ; shift
module load globus-cli/3.35.2 # load the globus cli sw
archer2_ep="3e90d018-0d05-461a-bbaf-aab605283d21"
jasmin_ep="a2f53b7f-1b4e-4dce-9b7c-349ae760fee0"
globus transfer $archer2_ep:$archer_dir $jasmin_ep:$jasmin_dir --recursive --sync-level checksum --preserve-timestamp --fail-on-quota-errors --verbose 
module unload globus-cli/3.35.2
