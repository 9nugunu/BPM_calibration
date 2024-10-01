import os, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import TestBench_data_processing as tb_processing

tb_processing.PlotSettings()

noamp_dir = "D:/Testbench_rawdata/240402-dynamic-range-direct/results/averaged_data/"
amp_dir = "D:/Testbench_rawdata/dynamic-range-240329/results/averaged_data/"

noAmpData = pd.read_csv(noamp_dir + "Averaged-data.csv", index_col=False)
Ampdata = pd.read_csv(amp_dir + "Averaged-data.csv", index_col=False)

noAmpData = noAmpData[(noAmpData.iloc[:,0] >= -60) & (noAmpData.iloc[:,0] != 15)]
Ampdata = Ampdata[Ampdata.iloc[:,0] >= -60]
# x_range = range(-97, -27, 5)
print(noAmpData)
plt.figure(1)

plot_color = ['r', 'b']

for i in range(1, len(Ampdata.columns)):
    if i == 1:
        plt.semilogy(Ampdata.iloc[:,0] - 7, Ampdata[f' {str(i)}Ch'], plot_color[0], label='w/ Amp.', marker='o')
    plt.semilogy(Ampdata.iloc[:,0] - 7, Ampdata[f' {str(i)}Ch'], plot_color[0])
    # plt.scatter(Ampdata.iloc[:,0] - 7, Ampdata[f' {str(i)}Ch'], c=plot_color[1])

for i in range(1, len(noAmpData.columns)):
    if i == 1:
        plt.semilogy(noAmpData.iloc[:,0] - 7, noAmpData[f' {str(i)}Ch'], c=plot_color[1], label='w/o Amp.', marker='s', linestyle='--')
    plt.semilogy(noAmpData.iloc[:,0] - 7, noAmpData[f' {str(i)}Ch'], c=plot_color[1], linestyle='--')
    # plt.scatter(noAmpData.iloc[:,0] - 7, noAmpData[f' {str(i)}Ch'], c=plot_color[0])



plt.title("Dynamic range of the read-out electronics", fontweight='bold')
plt.legend(loc='upper left', fontsize=14)
plt.xticks(noAmpData.iloc[:,0][::2] - 7)
# plt.title("Induced signal from electrodes")
plt.xlabel("Input S/G strength [dBm]")
plt.ylabel("RMSE channel strength [A.U.]")
plt.grid()
plt.savefig(
        f"sg_elec.png",
        format="png",
        dpi=500,
        bbox_inches="tight"
)
# plt.axvline(10.075, color='gray', linestyle='--')
# plt.axvline(0.075, color='gray', linestyle='--')
plt.show()

# plt.scatter(input_strength, amp_adc_data[f'adc{i}'][f' {i}Ch'], c='r')
# # elif i == 4:

# plt.semilogy(input_strength, amp_adc_data[f'adc{i}'][f' {i}Ch'], 'r')
# plt.scatter(input_strength, amp_adc_data[f'adc{i}'][f' {i}Ch'], c='r')