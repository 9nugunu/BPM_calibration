import pandas as pd
import os
import glob, re
import matplotlib.pyplot as plt
import numpy as np
import TestBench_data_processing as tb_dataprocessing
print(os.getcwd())

file_dir = './231215_dynamic_range/'
os.chdir("../" + file_dir)
print(os.getcwd())

raw_adc_files = glob.glob("./Electronics_dynamic range/ch*-adc*_dynamic.csv")
amp_adc_files = glob.glob("./Electronics_dynamic range/ch*-adc*_dynamic-amp.csv")

# 결과를 저장할 딕셔너리를 초기화합니다.
raw_adc_data = {}
amp_adc_data = {}

# \d: 숫자, +: 문자
pattern = re.compile(r'ch(\d+)-adc(\d+)_dynamic')

def sort_by_adc_number(file_path):
    match = pattern.search(file_path)
    if match:
        return int(match.group(2))
    return 0  # 패턴과 일치하지 않는 경우

# 파일을 ADC 번호에 따라 정렬합니다. # CHANGED
raw_adc_files_sorted = sorted(raw_adc_files, key=sort_by_adc_number)
amp_adc_files_sorted = sorted(amp_adc_files, key=sort_by_adc_number)

# 각 파일 쌍에 대해 처리를 반복합니다.
for raw_file, amp_file in zip(raw_adc_files_sorted, amp_adc_files_sorted):
    # 파일 이름에서 채널 번호를 추출합니다.
    match_raw = pattern.search(raw_file)
    match_amp = pattern.search(amp_file)
    if match_raw and match_amp:
        adc_key = f'adc{match_raw.group(2)}'  # adc 번호를 키로 사용합니다. # CHANGED
    
        # 각 파일을 DataFrame으로 로드합니다.
        raw_adc_df = pd.read_csv(raw_file, index_col='No')
        amp_adc_df = pd.read_csv(amp_file, index_col='No')
        
        # 추출한 ADC 번호를 키로 사용하여 딕셔너리에 DataFrame을 저장합니다.
        raw_adc_data[adc_key] = raw_adc_df
        amp_adc_data[adc_key] = amp_adc_df

# print(raw_adc_data)
# print("="*300)
# print(amp_adc_data)

tb_dataprocessing.PlotSettings()
input_strength = np.arange(-50, 21, 5)
plt.figure(1)

for i in range(1, 5):
    if i == 1:  # 첫 번째 시리즈에만 레이블을 지정합니다.
        plt.semilogy(input_strength[:-5], amp_adc_data[f'adc{i}'][f' {i}Ch'], 'r', label='w/ 30dB amp')
    else:
        plt.semilogy(input_strength[:-5], amp_adc_data[f'adc{i}'][f' {i}Ch'], 'r')
        
for i in range(1, 5):
    if i == 1:  # 첫 번째 시리즈에만 레이블을 지정합니다.
        plt.semilogy(input_strength, raw_adc_data[f'adc{i}'][f' {i}Ch'], 'b', label='w/o amp')
    else:
        plt.semilogy(input_strength, raw_adc_data[f'adc{i}'][f' {i}Ch'], 'b')

# amp_adc_data에 대한 데이터 시리즈를 그리고, 첫 번째 시리즈에만 레이블을 추가합니다.

plt.title("Connecting directly S/G to electronics")
plt.legend(loc='upper left')
plt.xlabel("Input amplitude [dBm]")
plt.ylabel("ADC count [A.U.]")

os.chdir("../")

plt.show()