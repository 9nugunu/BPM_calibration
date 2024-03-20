import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import TestBench_data_processing as tb

tb.PlotSettings()

print(os.getcwd())
file_dir = '\-5_5_dataset'
os.chdir("../" + file_dir)

samples = "2"
# df = pd.read_csv(f'plot_digitized_onlyData_{samples}.csv', index_col='Time[ns]')
df = pd.read_csv(f'No-signal-rms.csv', index_col='Time[ns]')
print(df)

graph_color = ('r', 'b', 'g', 'm')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

for i in range(0,1):
    # ax1.plot(df.index* 1e-3, df.iloc[:,i], lw=0.5, c=graph_color[i], label=f"{i+1} Ch")
    ax1.scatter(df.index* 1e-3, df.iloc[:,i], c=graph_color[i], s=10)

ax1.set_title('Digitized waveform generated by the read-out electronics', fontweight='bold')
ax1.set_xlabel('Time [\u03bcs]')
ax1.set_ylabel('ADC Count [A.U.]')
ax1.legend(loc='upper left', framealpha=0.95)

fft_results = {}
frequencies = np.fft.fftfreq(df.index.size, d=(df.index[1] - df.index[0]) * 1e-3) # Convert time to seconds
positive_freqs = np.abs(frequencies) <= 500 # frequencies > 0
target_freq = 352
ax2.axvline(x=target_freq, color='gray', linestyle='--')

for column in df.columns:
    fft_results[column] = np.fft.fft(df[column])

for i, (channel, fft_result) in enumerate(fft_results.items()):
    # ax2.plot(frequencies[positive_freqs], np.abs(fft_result[positive_freqs]), label=f"{i+1} Ch", c=graph_color[i])
    ax2.scatter(frequencies[positive_freqs], np.abs(fft_result[positive_freqs]), label=f"{i+1} Ch", c=graph_color[i])

    if np.any((frequencies > target_freq - 1) & (frequencies < target_freq + 1)):
        # Find the FFT result closest to the target frequency
        target_index = np.argmin(np.abs(frequencies - target_freq))
        target_magnitude = np.abs(fft_result[target_index])

        # # Add a marker at the target frequency
        # ax2.scatter(frequencies[target_index], target_magnitude, s=30, c=graph_color[i])


ax2.set_title('FFT of the digitized waveforms', fontweight='bold', fontsize=22)
ax2.set_xlabel('Frequency [MHz]')
ax2.set_ylabel('FFT Magnitude [A.U.]')
# ax2.set_ylim([0, 0.1*1e6])
# ax2.set_xticks([0, 100, 200, 300, 352, 400, 500])
# ax2.set_xlim([0, 1000])
ax2.legend(loc='upper left')

plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.savefig(f"RAW_FFT_{target_freq}MHz-{samples}samples.png",
        format="png",
        dpi=500,
        bbox_inches="tight",)
plt.show()
