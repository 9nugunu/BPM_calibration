import os
import pandas as pd
import numpy as np

def calculate_mean_std(data_dir, chs, x_point, y_point):
    file_list = os.listdir(data_dir)
    print(file_list)
    results = {}

    for ch in chs:
        results[ch] = []
        
    for file_name in file_list:
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            df = pd.read_csv(file_path)

            # filtering
            filtered_data = df[(df['x'] == x_point) & (df['y'] == y_point)]
            if not filtered_data.empty:
                for ch in results.keys():
                    results[ch].extend(filtered_data[ch].tolist())
    
    for ch, values in results.items():
        if values:
            mean_value = np.mean(values)
            std_dev = np.std(values)
            print(f"{ch} - (x,y) = ({x_point},{y_point}) mean: {mean_value:.4f} mm, sigma: {std_dev*1e3:.4f} μm")
        else:
            print(f"{ch} - (x,y) = ({x_point},{y_point}) 에 해당하는 데이터가 없습니다.")

def electronics_mean_std(data_dir, target_column, chs):
    file_list = os.listdir(data_dir)
    print(file_list)
    # MODIFICATION: Change results to store data grouped by input power
    results = {}

    # Initialize results dictionary with channels as keys and dictionaries as values
    for ch in chs:
        results[ch] = {}

    # Loop over each file in the directory
    for file_name in file_list:
        if file_name.lower().endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            df = pd.read_csv(file_path)

            # MODIFICATION: Iterate over each input power and collect channel data
            if not df.empty:
                for _, row in df.iterrows():
                    input_power = row[target_column]
                    if input_power not in results[ch]:
                        for ch in chs:
                            results[ch][input_power] = []
                    for ch in chs:
                        results[ch][input_power].append(row[ch])

    # Calculate the mean and standard deviation for each channel and input power
    # MODIFICATION: Print the results grouped by input power
    for ch, power_groups in results.items():
        print(f"Channel: {ch}")
        for input_power, values in power_groups.items():
            if values:
                mean_value = np.mean(values)
                std_dev = np.std(values)
                print(f"    Input Power {input_power} dBm - Mean: {mean_value:.4f}, STD: {std_dev:.4f}")
            else:
                print(f"    Input Power {input_power} dBm - No data available.")

data_dir = "C:/Users/9nugu/Documents/03-04-combined/results/5000 samples/Calcul_single_results/"#"D:/dynamic-range-240329/results"
# data_dir = "D:/Testbench_rawdata/240402-dynamic-range-direct/results"#"D:/dynamic-range-240329/results"
target_column = "Input power [dBm]"
dynamic_ch = [" 1Ch", " 2Ch", " 3Ch", " 4Ch"]
# 'D:/240401-03-rawdata-elec01/results/Calcul_single_results/'

dos_channels = [' X(A)', ' Y(A)'] + ['DOS_{equal}_X(A)_1', 'DOS_{equal}_Y(A)_1']
# dynamic_ch = [' 1Ch',' 2Ch',' 3Ch',' 4Ch']

std_list = []
x_point, y_point = 0,0
calculate_mean_std(data_dir, dos_channels, x_point, y_point)
# electronics_mean_std(data_dir, target_column, dynamic_ch)
