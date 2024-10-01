import pandas as pd
import matplotlib.pyplot as plt
import TestBench_data_processing as tb

tb.PlotSettings()
# 파일 경로
before_cal_path = "D:/Testbench_rawdata/240423-dynamic-amp-7dbm-beforeCal/results/averaged_data/Averaged-data_with_STD-data.csv"
after_cal_path = "D:/Testbench_rawdata/240425-dynamic-amp-7dbm/results/averaged_data/Averaged-data_with_STD-data.csv"
# after_cal_path = "D:/Testbench_rawdata/dynamic-range-240329/results/averaged_data/Averaged-data_with_STD-data.csv"

# 데이터 불러오기
before_data = pd.read_csv(before_cal_path)
after_data = pd.read_csv(after_cal_path)

# 그래프 초기화
fig, ax1 = plt.subplots(figsize=(12,8))

# 주축(ax1)에 평균값(Mean) 플롯 - before calibration
color = 'tab:blue'
ax1.set_xlabel('Input Power [dBm]')
ax1.set_ylabel('Channel RMSE strength [A.U.]')
ax1.semilogy(before_data['Input power [dBm]'][4:], before_data['Channel Mean'][4:], color=color, label='Before Cal')
# ax1.tick_params(axis='y', labelcolor=color)

# 보조축(ax2)에 표준편차(STD) 플롯 - before calibration
ax2 = ax1.twinx()  
# color = 'b'
ax2.set_ylabel('Channel STD [A.U.]')  # 보조축 레이블
ax2.set_yticks(range(0, 201, 25))
ax2.set_ylim([0, 200])
ax2.bar(before_data['Input power [dBm]'][4:], before_data['Channel STD'][4:], color=color, hatch="//", label='STD (Before Cal)')

# ax2.tick_params(axis='y', labelcolor=color)

color = 'tab:red'
ax1.semilogy(after_data['Input power [dBm]'][4:], after_data['Channel Mean'][4:], color=color, linestyle='--', label='After Cal')
ax2.bar(after_data['Input power [dBm]'][4:] + 0.4, after_data['Channel STD'][4:], color=color, width=0.4, label='STD (After Cal)')

plt.title('Analog devices calibration results')

ax1.legend(loc='upper left')
ax2.legend(loc='center right')

plt.show()
