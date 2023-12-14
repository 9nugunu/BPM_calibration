import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import TestBench_data_processing as tb_dataprocessing
print(os.getcwd())

tb_dataprocessing.PlotSettings()


Port = '4port/'
Wanted_data_x = ' X(C)'
Wanted_data_y = ' Y(C)'

FileName = 'cal_paper__' + '1' + '_4port_01_0.25.csv'
file_dir = './-5_5_dataset/' + Port + 'FOR_PAPER/' + FileName # 'PAPER_ONLY_0825/' +
data = pd.read_csv(file_dir, index_col=False)

data.drop([' Time', ' Type', ' 1Ch', ' 2Ch',  ' 3Ch', ' 4Ch', ' X(A)', ' X(B)', ' Y(A)', ' Y(B)'], axis=1, inplace=True)
# number_interval, data['x'], data['y'] = add_col_axis(data, step, measure_range)

print(data.head())

plt.scatter(data[Wanted_data_x], data[Wanted_data_y])
# plt.yticks([0.4, 0.3, 0.2, 0.1, 0.0, -0.1])
plt.title('measured raw data')
plt.xlabel('X raw data')
plt.ylabel('Y raw data')
# plt.savefig(file_dir+'RAW.png',
#     format='png',
#     dpi=1000,
# bbox_inches='tight')
plt.show()