import os, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import TestBench_data_processing as tb_dataprocessing

tb_dataprocessing.PlotSettings()

def calculate_rmse(df):
    rmse_values = {}
    for column in df.columns[1:]:  # 'Type' 열을 제외하고 각 채널에 대해 반복
        mean_value = df[column].mean()
        rmse = np.sqrt(((df[column] - mean_value) ** 2).mean())
        rmse_values[column] = rmse
    return rmse_values

dir_name = "C:/Users/9nugu/Documents/240327-electronicsTest/"
file_dir = (dir_name)
file_list = os.listdir(file_dir)
print(file_list)

columns_to_convert = [' 1Ch', ' 2Ch', ' 3Ch', ' 4Ch']
rmse_list = []
for i in file_list:
    data_path = os.path.join(file_dir, i)
    raw_data = pd.read_csv(data_path, low_memory=False) #index_col = False
    raw_data.drop([" X(A)", " Y(A)", " X(B)", " Y(B)", " X(C)", " Y(C)", " X(D)", " Y(D)"], axis=1, inplace=True)
    only_adc = raw_data.iloc[3:]
    for col in columns_to_convert:
        only_adc.loc[:,col] = only_adc.loc[:, col].astype(float)

    rmse_values = calculate_rmse(only_adc)
    rmse_list.append(rmse_values)

all_rmse_df = pd.DataFrame(rmse_list)
all_rmse_df['X(A)'] = all_rmse_df[' 2Ch'] - all_rmse_df[' 4Ch']
all_rmse_df['Y(A)'] = all_rmse_df[' 1Ch'] - all_rmse_df[' 3Ch']
print(all_rmse_df)

all_rmse_df.to_csv('calculated_rmse.csv', index=False)
