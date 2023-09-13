#!/bin/bash

## ----- No need to un-comment if raw data is not available
## generate unprocessed data for different entities in data
#filePrefixesToLoad=("apollo" "hazel" "lincoln_hazel" "human" "empty")
#for str in ${filePrefixesToLoad[@]}; do
#    python combo_no_label.py -m  $str
#done
#
## assemble data into train and test sets
#python assemble_data.py
#
## signal and pre-process data
#python signal_preprocessing.py


## ---------------------------------------------------

# train DCN

python dcn/dcn_house.py

## visualize/report results
python reports.py
