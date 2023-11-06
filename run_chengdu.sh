#!/bin/bash

# cd ../../priv_traj_gen

# dataset=chengdu
# max_size=10000
# data_name=${max_size}
# latlon_config=chengdu.json
# location_threshold=200
# time_threshold=60
# n_bins=30
# seed_for_dataset=0
# training_data_name=${location_threshold}_${time_threshold}_bin${n_bins}_seed${seed_for_dataset}

# python3 make_raw_data.py --original_data_name $dataset --max_size $max_size --seed $seed_for_dataset --save_name $data_name --n_bins $n_bins
# python3 data_pre_processing.py --latlon_config  $latlon_config --dataset $dataset --data_name $data_name --location_threshold $location_threshold --time_threshold $time_threshold --save_name $training_data_name --n_bins $n_bins $option --seed $seed_for_dataset

# location_threshold=0
# time_threshold=0
# route_data_name=${location_threshold}_${time_threshold}_bin${n_bins}_seed${seed_for_dataset}

# python3 data_pre_processing.py --latlon_config  $latlon_config --dataset $dataset --data_name $data_name --location_threshold $location_threshold --time_threshold $time_threshold --save_name $route_data_name --n_bins $n_bins $option --seed $seed_for_dataset

# cd ../MTNet_Code/MTNet

dataset=chengdu
max_size=10000
# n_bins=30
seed=0
# training_data_name=200_60_bin30_seed0
# route_data_name=0_0_bin30_seed0
training_data_dir=/data/${dataset}/${max_size}/MTNet
original_data_dir=/data/${dataset}/raw
# route_data_dir=/data/${dataset}/${max_size}/${route_data_name}
# stay_point_data_dir=/data/${dataset}/${max_size}/${training_data_name}
save_path=/data/results/${dataset}/${max_size}/MTNet
# setting_path=/data/${dataset}/${max_size}/0_0_bin${n_bins}_seed0/params.json
epoch=500
cuda_number=0
dp=False
python3 make_training_data.py $original_data_dir $training_data_dir $dataset $max_size $seed
python3 train.py $training_data_dir $save_path $epoch $cuda_number $dp 
# python3 convert_to_original_format.py $training_data_dir $save_path
# python3 evaluate.py $save_path $route_data_dir $stay_point_data_dir $save_path