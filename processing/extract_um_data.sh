# so have python env then do source processing/extract_um_data.sh
# give it at least two arguments -- directory to process and where you want output to go.
# runs on jasmin which is where AMIP data got transfered too.

in_dir=$1 ; shift
output_dir=$1/$(basename $in_dir) ; shift
echo "dir: $dir"
echo "output_dir: $output_dir"
echo "other args: $*"
# setup the env...
. $OPT_UKESM_ROOT/setup_jasmin # not sure why I need to do this
# any remaining args get passed through to extract_um_data
for dir in ${in_dir}/*Z
do
    output=${output_dir}/"$(basename ${dir}).nc"
    files=("${dir}"/*a.p[5m]*.pp)
    #echo "${files[@]} -> ${output}"
    cmd="extract_um_data.py ${files[@]} --output ${output} --log_level INFO --select_file ${OPT_UKESM_ROOT}/processing/select.json $* " # pass in all arguments"
    echo  "cmd: ${cmd}"

    ${cmd} # now run the cmd

done
