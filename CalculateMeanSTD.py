import os
import pandas as pd
import numpy as np

def calculate_mean_std(data_dir, x_point, y_point):
    file_list = os.listdir(data_dir)
    results = {}
    
    channels = [' 1Ch', ' 2Ch', ' 3Ch', ' 4Ch', ' X(A)', ' Y(A)']
    for channel in channels:
        results[channel] = []
        
    for file_name in file_list:
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            df = pd.read_csv(file_path)

            # filtering
            filtered_data = df[(df['x'] == x_point) & (df['y'] == y_point)]
            if not filtered_data.empty:
                for channel in results.keys():
                    results[channel].extend(filtered_data[channel].tolist())
    
    for channel, values in results.items():
        if values:  # 값이 있을 경우에만 계산
            mean_value = np.mean(values)
            std_dev = np.std(values)
            print(f"{channel} - (x,y) = ({x_point},{y_point}) mean: {mean_value:.4f}, sigma: {std_dev:.4f}")
        else:
            print(f"{channel} - (x,y) = ({x_point},{y_point}) 에 해당하는 데이터가 없습니다.")


data_dir = 'D:/240331-0dbm-rawdata/results/'  # 데이터 파일이 저장된 디렉토리 경로
x_point, y_point = 10, 10

calculate_mean_std(data_dir, x_point, y_point)
