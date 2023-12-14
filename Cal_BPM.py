import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import TestBench_data_processing as tb_dataprocessing
print(os.getcwd())

tb_dataprocessing.PlotSettings()

number_interval = 21

step = 0.5
max_point = 5
Port = '4port/'
Wanted_data = {'X': ' X(C)', 'Y': ' Y(C)'}

filename = 'cal_paper__' + '1' + '_4port_01_0.25.csv'
file_dir = './-5_5_dataset/' + Port + 'FOR_PAPER/' #+ filename # 'PAPER_ONLY_0825/' +
os.chdir('../' + file_dir)
print(os.getcwd())

data = pd.read_csv(filename, index_col=False)

data.drop([' Time', ' Type', ' 1Ch', ' 2Ch',  ' 3Ch', ' 4Ch', ' X(A)', ' X(B)', ' Y(A)', ' Y(B)'], axis=1, inplace=True)
data['x'], data['y'] = tb_dataprocessing.add_col_axis(number_interval, step, max_point)

# print(data.head())

plt.scatter(data[Wanted_data['X']], data[Wanted_data['Y']])
# plt.yticks([0.4, 0.3, 0.2, 0.1, 0.0, -0.1])
plt.title('measured raw data')
plt.xlabel('X raw data')
plt.ylabel('Y raw data')
# plt.savefig(file_dir+'RAW.png',
#     format='png',
#     dpi=1000,
# bbox_inches='tight')
# plt.show()

cal_offset = data[(data['x'] == 0) & (data['y'] == 0)][[Wanted_data['X'], Wanted_data['Y']]]
x_offset = cal_offset[' X(C)'].values[0]*1e3
y_offset = cal_offset[' Y(C)'].values[0]*1e3
print(fr"x_offset: {x_offset} μm")
print(f"y_offset: {y_offset} μm")

mean_same_x = data.groupby('x').mean()
mean_same_y = data.groupby('y').mean()

# print(mean_same_x)
# plt.figure(2)
# plt.scatter(data[data['x'] == data['y']]['x'], data[data['x'] == data['y']][Wanted_data['X']], label='on_axis')
# plt.scatter(mean_same_x.index, mean_same_x[' X(C)'], label='mean_same_x')
# plt.legend()

# plt.figure(3)
# plt.scatter(data[data['x'] == data['y']]['y'], data[data['x'] == data['y']][Wanted_data['Y']], label='on_axis')
# plt.scatter(mean_same_y.index, mean_same_y[' Y(C)'], label='mean_same_y')
# plt.legend()

cal_x, cal_y = tb_dataprocessing.optimized_func(data, Wanted_data, max_point)
# cal_x_dia, cal_y_dia = optimized_func(data['xDia'], data['yDia'])
data['cal_X'], data['cal_Y']  = cal_x, cal_y

x_dummy = np.arange(-5, 5.1, 0.5)
y_dummy = x_dummy
mean_same_x = data.groupby('x').mean()['cal_X']
mean_same_y = data.groupby('y').mean()['cal_Y']
print(mean_same_x)
plt.figure(4)
plt.plot(x_dummy, y_dummy, c='k',  lw=1, ls='-')
plt.scatter(mean_same_x.index, mean_same_x.values, lw=0.8, marker='^', facecolor='none', edgecolor='b')
plt.scatter(mean_same_y.index, mean_same_y.values, marker=',', facecolor='none', edgecolors='r')
plt.title("Linear calibration result")
plt.xlabel("Wire position [mm]")
plt.ylabel("Linear Estimation [mm]")
plt.gca().set_aspect('equal')
# plt.ylabel("K$_{x, y}$ X DOS ($\Delta/\Sigma$)")
plt.grid()
plt.show()

