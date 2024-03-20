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

print(f"raw_adc_files: {raw_adc_files}")
print(f"amp_adc_files: {amp_adc_files}")
raw_adc_data = {}
amp_adc_data = {}

# \d: 숫자, +: 문자
pattern = re.compile(r'ch(\d+)-adc(\d+)_dynamic')

def sort_by_adc_number(file_path):
    match = pattern.search(file_path)
    if match:
        return int(match.group(2))
    return 0  # 패턴과 일치하지 않는 경우

raw_adc_files_sorted = sorted(raw_adc_files, key=sort_by_adc_number)
amp_adc_files_sorted = sorted(amp_adc_files, key=sort_by_adc_number)

# print(f"raw_adc_files: {raw_adc_files_sorted}")
# print(f"amp_adc_files: {amp_adc_files_sorted}")

for raw_file, amp_file in zip(raw_adc_files_sorted, amp_adc_files_sorted):
    print(raw_file)
    # 채널 번호 추출
    match_raw = pattern.search(raw_file)
    match_amp = pattern.search(amp_file)
    if match_raw and match_amp:
        adc_key = f'adc{match_raw.group(2)}'  # adc 번호를 키로 사용

        raw_adc_df = pd.read_csv(raw_file, index_col='No')
        amp_adc_df = pd.read_csv(amp_file, index_col='No')
        print(amp_adc_df.index)
            # amp_adc_df.set_index(raw_adc_df.columns.values.tolist()[0])
            # raw_adc_df.set_index(raw_adc_df.columns.values.tolist()[0])
        # amp_adc_df.set_index('No', inplace=True)
        print(raw_adc_data)
        raw_adc_data[adc_key] = raw_adc_df
        raw_adc_data[adc_key].drop([" Time", " Type", " X(A)", " Y(A)", " X(B)", " Y(B)", " X(C)", " Y(C)", " X(D)", " Y(D)"], axis=1, inplace=True)
        # print(raw_adc_data[adc_key])
        amp_adc_data[adc_key] = amp_adc_df
        amp_adc_data[adc_key].drop([" Time", " Type", " X(A)", " Y(A)", " X(B)", " Y(B)", " X(C)", " Y(C)", " X(D)", " Y(D)"], axis=1, inplace=True)


# for i in range(1, 5):
#     print(raw_adc_data[f'adc{i}'])
# print("="*300)
# # print(amp_adc_data)

tb_dataprocessing.PlotSettings()
plt.figure(figsize=(7, 5))

for i in range(1, 5):
    if i == 1:
        input_strength = np.arange(-80, -19, 5)
        plt.semilogy(input_strength, amp_adc_data[f'adc{i}'][f' {i}Ch'], 'r', label='W/ 30dB Amp.')
        plt.scatter(input_strength, amp_adc_data[f'adc{i}'][f' {i}Ch'], c='r')
    # elif i == 4:

    else:
        plt.semilogy(input_strength, amp_adc_data[f'adc{i}'][f' {i}Ch'], 'r')
        plt.scatter(input_strength, amp_adc_data[f'adc{i}'][f' {i}Ch'], c='r')
        
for i in range(1, 5):
    if i == 1:
        input_strength = np.arange(-80, 11, 5)
        plt.semilogy(input_strength, raw_adc_data[f'adc{i}'][f' {i}Ch'], 'b', label='W/o Amp.')
        plt.scatter(input_strength, raw_adc_data[f'adc{i}'][f' {i}Ch'], c='b')
    # elif i == 4:

    else:
        plt.plot(input_strength, raw_adc_data[f'adc{i}'][f' {i}Ch'], 'b')
        plt.scatter(input_strength, raw_adc_data[f'adc{i}'][f' {i}Ch'], c='b')

plt.title("Dynamic range of the read-out electronics", fontweight='bold')
plt.legend(loc='upper left')
plt.xticks(input_strength[::2])
plt.xlabel("Input power [dBm]")
plt.ylabel("ADC count [A.U.]")
plt.grid()
# os.chdir("../")

plt.savefig(
        f"sg_elec.png",
        format="png",
        dpi=500,
        bbox_inches="tight"
)

plt.show()