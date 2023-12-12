import os
import pandas as pd
import numpy as np

print(os.getcwd())

Port = '4port/'

FileName = 'cal_paper__' + '1' + '_4port_01_0.25.csv'
file_dir = './-5_5_dataset/' + Port + 'FOR_PAPER/' + FileName # 'PAPER_ONLY_0825/' +
data = pd.read_csv(file_dir, index_col=False)

data[' X(E)'] = np.log(data[' 2Ch'] * data[' 3Ch']) - (np.log(data[' 1Ch'] * data[' 4Ch']))
data[' Y(E)'] = np.log(data[' 2Ch'] * data[' 1Ch']) - (np.log(data[' 3Ch'] * data[' 4Ch']))
data.head()
data.drop([' Time', ' Type', ' 1Ch', ' 2Ch',  ' 3Ch', ' 4Ch', ' X(A)', ' X(B)', ' Y(A)', ' Y(B)'], axis=1, inplace=True)
# number_interval, data['x'], data['y'] = add_col_axis(data, step, measure_range)

print(data.head())