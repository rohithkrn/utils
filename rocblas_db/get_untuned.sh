#!/bin/bash

# reads in rocblas bench commands from
# ROCBLAS_LAYER=2 logs and verfies if 
# tuned? with TENSILE_DB=2 and saves a 
# a separate log file for each bench 
# command and appends the bench command 
# at the end of the file

x=1
export TENSILE_DB=2
export HIP_VISIBLE_DEVICES=0

while IFS= read -r line; do
    echo $line
    $line |& tee logs/${x}.txt	
    echo "$line" >> logs/${x}.txt
    echo "Wrote ${x}.txt"
    x=`expr $x + 1`
done < ./rblas.txt

#check if tuned
#ls ./logs | while read line; do echo $line; grep "distance=0" logs/$line; done;
echo "============printing untuned files and write to filter.txt================="
for file in $(ls logs); do
    grep -q "distance=0" logs/${file}
    if [ $? != 0 ]
    then
        echo $file |& tee  "filter.txt"
    fi
done;

