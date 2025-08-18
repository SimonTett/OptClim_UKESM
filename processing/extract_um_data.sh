# so have python env then do source processing/extract_um_data.sh
for dir in /gws/nopw/j04/terrafirma/tetts/um_archive/u-dr157/*Z
do
    output=${dir}".nc"
    files=("${dir}"/*a.p[5m]*.pp)
    echo "${files[@]} -> ${output}"
    extract_um_data.py ${files[@]} --output ${output} --log_level DEBUG --select_file processing/select.json
done
