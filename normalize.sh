for dir in ./1005s/*/
do
    dir=${dir/}
    echo $dir
    ./normalize_dir.wls $dir
done
