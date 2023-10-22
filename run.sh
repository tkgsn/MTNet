save_path=/data/results/geolife/0/narrow_0_0_bin30_seed0/MTNet
original_data_path=/data/geolife/0/narrow_0_0_bin30_seed0/training_data.csv
original_data_dir=/data/geolife/0/narrow_0_0_bin30_seed0
python3 make_training_data.py $original_data_path $save_path
python3 train.py
python3 convert_to_original_format.py $save_path $save_path
python3 evaluate.py $save_path $original_data_dir $save_path