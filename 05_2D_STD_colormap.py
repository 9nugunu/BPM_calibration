import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import TestBench_data_processing as tb_processing

tb_processing.PlotSettings()

def calculate_std_maps(data_dir, x_range, y_range, step=1):
    x_values = np.arange(-x_range, x_range + step, step)
    y_values = np.arange(-y_range, y_range + step, step)
    std_map_x = np.zeros((len(y_values), len(x_values)))
    std_map_y = np.zeros((len(y_values), len(x_values)))
    
    for ix, x in enumerate(x_values):
        for iy, y in enumerate(y_values):
            diff_x_list, diff_y_list = [], []
            for file_name in os.listdir(data_dir):
                if file_name.lower().endswith('.csv'):
                    df = pd.read_csv(os.path.join(data_dir, file_name))
                    filtered_data = df[(df['x'] == x) & (df['y'] == y)]
                    if not filtered_data.empty:
                        diff_x = filtered_data['DOS_{all}_X(A)_3']
                        diff_y = filtered_data['DOS_{all}_Y(A)_3']
                        diff_x_list.extend(diff_x.tolist())
                        diff_y_list.extend(diff_y.tolist())
            if diff_x_list:
                std_map_x[iy, ix] = np.std(diff_x_list)
            if diff_y_list:
                std_map_y[iy, ix] = np.std(diff_y_list)
    
    combined_std_map = np.sqrt(std_map_x**2 + std_map_y**2)
    print(f"유클라디안 STD : {np.max(combined_std_map)*1e3} μm")
    return x_values, y_values, std_map_x, std_map_y
data_dir = "C:/Users/9nugu/Documents/03-04-combined/results/5000 samples/Calcul_single_results/"
# data_dir = "C:\\Users\\9nugu\\Google_Drive\\내 드라이브\\Graduate\\1. BPM\\1. Projects\\1. BPM Test Bench\\실험데이터\\-5_5_dataset\\2port\\BPM01_352MHz_variAmp\\results\\"
# 사용 예:
x_range, y_range = 8, 8
x_values, y_values, std_map_x, std_map_y = calculate_std_maps(data_dir, x_range, y_range)
vmax = 150
print(f"max_std_x: {np.max(std_map_x)*1e3}, max_std_y: {np.max(std_map_y)*1e3}")
# print(f"유클라디안 STD {round(np.max(np.sqrt((std_map_x*1e3)**2 + (std_map_y*1e3)**2),2))}")
# std_map_x와 std_map_y 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
X, Y = np.meshgrid(x_values, y_values)

cs1 = ax1.contourf(X, Y, std_map_x*1e3, 40, cmap='jet', vmin=0, vmax=vmax)
# fig.colorbar(cs1, ax=ax1, orientation='vertical')
# cbar1 = fig.colorbar(cs1, ax=ax1, orientation='vertical')
# cbar1.set_ticks([0, 0.25, 0.5, 0.75, 1.0])  # 컬러바의 눈금 설정
# cbar1.set_label('STD of X(A)', rotation=270, labelpad=20)  # 컬러바 레이블 설정
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.1)
cbar1 = mpl.colorbar.ColorbarBase(cax1, cmap=cs1.cmap, norm=cs1.norm)
# cbar2 = fig.colorbar(cs2, ax=ax2, orientation='vertical')
cbar1.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
cbar1.set_ticks(range(0,vmax+1,30))
ax1.set_title('X STD of each measurement', fontweight='bold')
ax1.set_xticks(range(-8, 9, 4))
ax1.set_yticks(range(-8, 9, 4))
ax1.set_xlabel('X [mm]')
ax1.set_ylabel('Y [mm]')

cs2 = ax2.contourf(X, Y, std_map_y*1e3, 40, cmap='jet', vmin=0, vmax=vmax)
# fig.colorbar(cs2, ax=ax2, orientation='vertical')
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.1)
cbar2 = mpl.colorbar.ColorbarBase(cax2, cmap=cs2.cmap, norm=cs2.norm)
# cbar2 = fig.colorbar(cs2, ax=ax2, orientation='vertical')
cbar2.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
cbar2.set_ticks(range(0,vmax+1,30))
# cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1.0])  # 컬러바의 눈금 설정
# cbar2.set_label('STD of Y(A)', rotation=270, labelpad=20)  # 컬러바 레이블 설정
ax2.set_title('Y STD of each measurement', fontweight='bold')
ax2.set_xticks(range(-8, 9, 4))
ax2.set_yticks(range(-8, 9, 4))
ax2.set_xlabel('X [mm]')
ax2.set_ylabel('Y [mm]')

plt.savefig(
        f"STD-2dcolormap.png",
        format="png",
        dpi=500,
        bbox_inches="tight"
)
plt.tight_layout()
# plt.show()
plt.close()
# cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.00])
# cbar.set_label("Error [mm]", rotation=270, labelpad=32)
# cs2 = ax1.contourf(cs, levels=[0.1], colors="yellow")
# plt.colorbar(label='STD')
# plt.xlabel('X [mm]')
# plt.ylabel('Y [mm]')
# plt.title('STD Colormap')
# plt.close()

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # 데이터 포인트의 위치 및 너비, 깊이, 높이 정의
# pos_x = X.flatten()
# pos_y = Y.flatten()
# pos_z = np.zeros_like(std_map.flatten())
# dx = dy = np.ones_like(std_map.flatten()) * (x_values[1] - x_values[0]) * 0.8  # 너비 및 깊이 설정
# dz = std_map.flatten()  # 높이는 STD 값

# ax.bar3d(pos_x, pos_y, pos_z, dx, dy, dz, color='skyblue')

# ax.set_xlabel('X [mm]')
# ax.set_ylabel('Y [mm]')
# ax.set_zlabel('STD')
# ax.set_title('3D STD Bar Graph')
