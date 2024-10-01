import os
import pandas as pd
import numpy as np
import concurrent.futures

def calculate_rmse(only_adc):
    # RMSE 계산의 벡터화된 버전
    mean_values = only_adc.mean()
    print(mean_values)
    print("8"*100)
    # for column in only_adc.columns[1:]:  # 'Type' 열을 제외하고 각 채널에 대해 반복
    #     mean_value = only_adc[column].mean()
    #     rmse = np.sqrt(((df[column] - mean_value) ** 2).mean())
    #     rmse_values[column] = rmse
    rmse_values = np.sqrt(((only_adc - mean_values) ** 2).mean())
    print(rmse_values)
    print("2"*100)
    return rmse_values

dir_name = "C:\\Users\\9nugu\\Documents\\03-04-combined\\"
file_dir = dir_name
file_list = [file for file in os.listdir(file_dir) if file.lower().endswith('.csv')]
file_list = sorted(file_list)
RMSE_chdata = pd.DataFrame()

# 데이터 변환을 위한 전처리 함수
def preprocess_data(file_name):
    data_path = os.path.join(file_dir, file_name)
    raw_data = pd.read_csv(data_path, skiprows=3)  # low_memory 옵션 생략
    # 필요한 열만 유지하고 나머지 제거
    raw_data = raw_data[[' 1Ch', ' 2Ch', ' 3Ch', ' 4Ch']][1:].astype(float)
    print(raw_data)
    rmse_values = calculate_rmse(raw_data)
    print(rmse_values)
    RMSE_chdata = pd.concat(rmse_values, ignore_index=True)
    print(RMSE_chdata)
    os._exit(1)
    return RMSE_chdata

# count = 0
# 병렬 처리를 위한 파일 읽기
with concurrent.futures.ThreadPoolExecutor() as executor:
    data_frames = list(executor.map(preprocess_data, file_list))

# 모든 데이터프레임을 하나로 결합
combined_data = pd.concat(data_frames, ignore_index=True)

# RMSE 계산
rmse_values = calculate_rmse(combined_data)

# 결과 출력
print(rmse_values)

# 결과 저장
rmse_values.to_csv(os.path.join(dir_name, 'RMSE_DOS-data.csv'), index=False)
