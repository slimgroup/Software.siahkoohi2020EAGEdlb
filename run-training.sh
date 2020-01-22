#!/bin/bash -l

experiment_name=imaging-and-UQ
repo_name=seismic-imaging-with-SGLD

path_script=$HOME/$repo_name/src
vel_dir=$HOME/$repo_name/vel_dir
mkdir -p $vel_dir

if [ ! -f $vel_dir/overthrust_model.h5 ]; then
	wget https://github.com/slimgroup/JUDI.jl/raw/master/data/overthrust_model.h5 \
		-O $vel_dir/overthrust_model.h5
fi

python $path_script/main.py --epoch 10000 --eta 2.3177 --lr 0.001 --experiment $experiment_name \
--weight_decay 170.0 --cuda 0 --save_freq 50 --sample_freq 50 --vel_dir $vel_dir