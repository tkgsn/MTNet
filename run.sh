python3 make_training_data.py /data/geolife/0/narrow_0_0_bin30_seed0/training_data.csv ./data/geolife
python3 train.py
python3 convert_to_original_format.py ./data/geolife ./data/geolife /data/results/geolife/0/narrow_200_10_bin30_seed0/MTNet