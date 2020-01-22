#!/bin/bash -l

experiment_name=imaging-and-UQ
repo_name=seismic-imaging-with-SGLD
path_script=$HOME/$repo_name/src

echo Expeiment name: $experiment_name
python $path_script/main.py --experiment $d --phase test --cuda 0
python $path_script/main.py --experiment $d --phase prior --cuda 1

