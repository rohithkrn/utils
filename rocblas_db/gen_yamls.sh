#!/bin/bash

# reads in filter.txt from get_untuned.sh
# and generates yaml file with ROCBLAS_LAYER=4
# for each config in yamls directory

export ROCBLAS_LAYER=4
export HIP_VISIBLE_DEVICES=0 
x=1
while IFS= read -r line; do
    #echo $line;
    export ROCBLAS_LOG_PROFILE_PATH="yamls/${x}.yaml"
    config=$(tail -n 1 logs/$line)
    echo $config
    $config 
    x=`expr $x + 1`
done < ./filter.txt    
