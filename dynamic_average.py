import os
import pandas as pd
import numpy as np

file_dir = "D:/Testbench_rawdata/240402-dynamic-range-direct/results/"
file_dir = "D:/Testbench_rawdata/240423-dynamic-amp-7dbm-beforeCal/results/"
file_dir = "D:/Testbench_rawdata/dynamic-range-240329/results/"
file_dir = "D:/Testbench_rawdata/240425-dynamic-amp-7dbm/results/"

save_dir = file_dir + "averaged_data"
file_list = os.listdir(file_dir)
file_list = [file for file in file_list if file.lower().endswith('.csv')]
print(file_list)
target_data = ["Input power [dBm]", " 1Ch", " 2Ch", " 3Ch", " 4Ch"]
data_sum = pd.DataFrame()
data_frames = []

# Collecting data
for file in file_list:
    data_path = os.path.join(file_dir, file)
    data = pd.read_csv(data_path, index_col=False)
    data_frames.append(data[target_data])

data_avg = pd.concat(data_frames).groupby("Input power [dBm]").mean().reset_index()

data_avg['Channel Mean'] = data_avg[target_data[1:]].mean(axis=1)
data_avg['Channel STD'] = data_avg[target_data[1:]].std(axis=1, ddof=0)
# Ensure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save to CSV
save_path = os.path.join(save_dir, 'Averaged-data_with_STD-data.csv')
data_avg.to_csv(save_path, index=False)
