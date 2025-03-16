#!/usr/bin/env bash

pyfile=$1
dir_cfg=$2
dataset=$3
type=$4
gpu=$5


# Check if the pyfile exists
if [ ! -f $file ]; then
  echo "File '$file' does not exist. Exiting."
  exit
fi

# Check if the dir_cfg exists
if [ ! -d $dir_cfg ]; then
  echo "Directory $dir_cfg does not exist. Exiting."
  exit
fi

if [ -z $dataset ]; then
    echo "\$dataset is empty. Exiting."
    exit
fi
if [ -z $type ]; then
    echo "\$type is empty. Exiting."
    exit
fi

# List the matched filenames
echo "Matched: "
for filename in "${dir_cfg}/${dataset}/${type}"/*
do
    echo $filename
done
echo

echo "Start training"
for filename in "${dir_cfg}/${dataset}/${type}"/*
do
    echo $filename
    python $pyfile --cfg $filename --gpu $gpu
done


# for dataset in office officehome
# do
#     for type in OPDA ODA PDA CDA
#     do
#         # List the matched filenames
#         echo "Matched: "
#         for filename in "${dir_cfg}/${dataset}/${type}"/*
#         do
#             echo $filename
#         done
#     done
# done
