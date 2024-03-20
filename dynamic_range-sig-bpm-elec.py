import pandas as pd
import os
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import TestBench_data_processing as tb_dataprocessing
print(os.getcwd())

tb_dataprocessing.PlotSettings()

file_dir = './231215_dynamic_range/'
os.chdir("../" + file_dir)
print(os.getcwd())

filename = []
for file in os.listdir():
    if fnmatch.fnmatch(file, '0*_1217.csv'):
        filename.append(file)

sorted_filename = sorted(filename)
input_strength = np.arange(-60, 21, 5)

plot_color = ['r', 'b']
added_legend_amp = False  # Flag to track legend entry for '_w_amp_' files
added_legend_other = False  # Flag to track legend entry for other files

for index, file in enumerate(sorted_filename):
    if fnmatch.fnmatch(file, '*Motor_10_10*'):
        data = pd.read_csv(file, index_col=False)
        for i in range(1, 5):
            data[f' {i}Ch'] = np.log(data[f' {i}Ch'])
            if fnmatch.fnmatch(file, '*_w_amp_*'):
                if not added_legend_amp:
                    plt.plot(input_strength, data[f' {i}Ch'], plot_color[0], label='With Amp')
                    added_legend_amp = True  # Update the flag
                else:
                    plt.plot(input_strength, data[f' {i}Ch'], plot_color[0])
            else:
                if not added_legend_other:
                    plt.plot(input_strength, data[f' {i}Ch'], plot_color[1], label='Without Amp')
                    added_legend_other = True  # Update the flag
                else:
                    plt.plot(input_strength, data[f' {i}Ch'], plot_color[1])
        plt.xticks(input_strength[::2])
        plt.title("Induced signal from electrodes")
        plt.xlabel("S/G Input power [dBm]")
        plt.ylabel("ADC count [A.U.]")
        plt.axvline(10.075, color='gray', linestyle='--')
        plt.axvline(0.075, color='gray', linestyle='--')
plt.legend()
plt.savefig(
        f"dynamic_10-10_bpm.png",
        format="png",
        dpi=1000,
        bbox_inches="tight"
)
plt.show()


# plt.show()

# raw_adc_data = {}
# amp_adc_data = {}

# # \d: 숫자, +: 문자
# pattern = re.compile(r'ch(\d+)-adc(\d+)_dynamic')

# def sort_by_adc_number(file_path):
#     match = pattern.search(file_path)
#     if match:
#         return int(match.group(2))
#     return 0  # 패턴과 일치하지 않는 경우

# raw_adc_files_sorted = sorted(raw_adc_files, key=sort_by_adc_number)
# amp_adc_files_sorted = sorted(amp_adc_files, key=sort_by_adc_number)

# for raw_file, amp_file in zip(raw_adc_files_sorted, amp_adc_files_sorted):
#     # 채널 번호 추출
#     match_raw = pattern.search(raw_file)
#     match_amp = pattern.search(amp_file)
#     if match_raw and match_amp:
#         adc_key = f'adc{match_raw.group(2)}'  # adc 번호를 키로 사용
    
#         raw_adc_df = pd.read_csv(raw_file, index_col='No')
#         amp_adc_df = pd.read_csv(amp_file, index_col='No')
        
#         raw_adc_data[adc_key] = raw_adc_df
#         amp_adc_data[adc_key] = amp_adc_df

# # print(raw_adc_data)
# # print("="*300)
# # print(amp_adc_data)

# tb_dataprocessing.PlotSettings()

# plt.figure(1)

# for i in range(1, 5):
#     if i == 1:
#         plt.semilogy(input_strength[:-5], amp_adc_data[f'adc{i}'][f' {i}Ch'], 'r', label='w/ 30dB amp')
#     else:
#         plt.semilogy(input_strength[:-5], amp_adc_data[f'adc{i}'][f' {i}Ch'], 'r')
        
# for i in range(1, 5):
#     if i == 1:
#         plt.semilogy(input_strength, raw_adc_data[f'adc{i}'][f' {i}Ch'], 'b', label='w/o amp')
#     else:
#         plt.semilogy(input_strength, raw_adc_data[f'adc{i}'][f' {i}Ch'], 'b')

# plt.title("Connecting directly S/G to electronics")
# plt.legend(loc='upper left')
# plt.xlabel("Input amplitude [dBm]")
# plt.ylabel("ADC count [A.U.]")

# os.chdir("../")

plt.show()