import os, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import TestBench_data_processing as tb_dataprocessing

tb_dataprocessing.PlotSettings()

def calculate_rmse(df):
    rmse_values = {}
    # method 1
    # for column in df.columns[1:]:  # 'Type' 열을 제외하고 각 채널에 대해 반복
    # method 2
    # mean_value = df.mean()
    # rmse = np.sqrt(((df - mean_value) ** 2).mean())
    # rmse_values = rmse
    # method 3
    rmse_values = ((only_adc - only_adc.mean()) ** 2).mean().apply(np.sqrt).to_dict()
    return rmse_values

# dir_name = "C:\\Users\\9nugu\\Documents\\03-04-combined\\"#"C:/Users/9nugu/Documents/240331-rawdata"#/test01"#dynamic-range-240329/"
dir_name = "D:/Testbench_rawdata/240425-dynamic-amp-7dbm/"#"C:/Users/9nugu/Documents/240331-rawdata"#/test01"#dynamic-range-240329/"
file_dir = (dir_name)
file_list = os.listdir(file_dir)
file_list = [file for file in file_list if file.lower().endswith('.csv')]
file_list = sorted(file_list)
# print(file_list[882:1323])
# os._exit(1)
columns_to_convert = [' 1Ch', ' 2Ch', ' 3Ch', ' 4Ch']
rmse_list = []
samples = 5000 + 1
for count, file_name in enumerate(file_list, 1):
    data_path = os.path.join(file_dir, file_name)
    raw_data = pd.read_csv(data_path, skiprows=3) #index_col = False
    # raw_data.drop([" X(A)", " Y(A)", " X(B)", " Y(B)", " X(C)", " Y(C)", " X(D)", " Y(D)"], axis=1, inplace=True)
    only_adc = raw_data[[' 1Ch', ' 2Ch', ' 3Ch', ' 4Ch']][1:samples].astype(float) # 첫번째 raw 데이터행까지 버림

    rmse_values = calculate_rmse(only_adc)
    rmse_list.append(rmse_values)

    print(f"{count} / {len(file_list)}")

all_rmse_df = pd.DataFrame(rmse_list)
all_rmse_df[' X(A)'] = (all_rmse_df[' 2Ch'] - all_rmse_df[' 4Ch'])/(all_rmse_df[' 2Ch'] + all_rmse_df[' 4Ch'])
all_rmse_df[' Y(A)'] = (all_rmse_df[' 1Ch'] - all_rmse_df[' 3Ch'])/(all_rmse_df[' 1Ch'] + all_rmse_df[' 3Ch'])
# print(all_rmse_df)

output_dir = os.path.join(dir_name, f"results/RMSE-table-{samples - 1}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

os.chdir(output_dir)
# reversed_all_rmse_df = all_rmse_df.iloc[::-1].reset_index(drop=True)
# print(reversed_all_rmse_df)
all_rmse_df.to_csv('RMSE_DOS-data.csv', index=False)
