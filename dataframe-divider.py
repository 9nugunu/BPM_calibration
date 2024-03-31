import pandas as pd
import numpy as np
import os

class DataFrameSlicer:
    '''
    작성자: 김근우
    작성일: 24. 03. 31.
    
    기능
     - 사전에 저장되어있는 임의의 Dataframe을 지정한 크기에 맞춰 행 분리
     - dynamic-range, dos 경우에 대해 상응되는 상수 행을 추가.
     - 분리한 dataframe 각각을 csv파일로 저장
    '''
    def __init__(self, dataframe, slice_size):
        self.dataframe = dataframe
        self.slice_size = slice_size
        self.sliced_data = {}
        self.case = ''

    def add_new_column(self, case='dynamic_range', **kwargs):
        self.case = case
        total_sets = len(self.dataframe) // self.slice_size
        if case == 'dynamic_range':
            self.dataframe.insert(0, 'Input power [dBm]', np.tile(np.arange(-90, -19, 5), total_sets))
        elif case == 'dos':
            max_point = kwargs.get('max_point', 10)
            step = kwargs.get('step', 1)
            number_interval = kwargs.get('number_interval', 21)
            repeated_x, y = self.add_col_axis(max_point, step, number_interval, len(self.dataframe))
            # print(repeated_x)
            self.dataframe['x'] = repeated_x[:len(self.dataframe)]
            self.dataframe['y'] = y[:len(self.dataframe)]
        else:
            print(f"Case '{case}' not recognized.")

    def add_col_axis(self, max_point, step, number_interval, total_length):
        x = [i for i in np.arange(max_point, -max_point - step, -step)]
        repeated_x = np.tile(x, (total_length // len(x)))
        y = np.tile(np.repeat(x, number_interval),self.slice_size)
        # print(repeated_x, y)
        return repeated_x, y
    
    def slice_dataframe(self):
        # slice_size 행 씩 분할저장
        for i in range(0, len(self.dataframe), self.slice_size):
            self.sliced_data[f'data_{str(i//self.slice_size).zfill(2)}'] = self.dataframe.iloc[i:i+self.slice_size]

    def save_slices_to_csv(self, save_path, input=''):
        os.chdir(save_path)
        for key, df_slice in self.sliced_data.items():
            if input != '':
                file_name = f"{self.case}_{input}_{key}.csv"
            else:    
                file_name = f"{self.case}_{key}.csv"
            df_slice.to_csv(file_name, index=False)
            print(f"Saved {file_name}")


choice_data = {0:'dynamic_range', 1:'DOS'}
selected = choice_data[1]

if selected == 'dynamic_range':
    '''
    파일 경로 및 불러오기
    '''
    dir_name = "D:/dynamic-range-240329/results/"#/test01"#dynamic-range-240329/"
    file_dir = (dir_name)
    file_name = "calculated_rmse-dy.csv"
    data_path = os.path.join(file_dir, file_name)
    target_data = pd.read_csv(data_path)
    n = 15

    '''
    클래스 인스턴스 생성
    '''
    save_path = dir_name
    slicer = DataFrameSlicer(target_data, slice_size=n)
    slicer.add_new_column(case='dynamic_range')
    # slicer.add_new_column(case='dos', max_point=10, step=1, number_interval=15)
    slicer.slice_dataframe()
    slicer.save_slices_to_csv(save_path)

elif selected == 'DOS':
    '''
    파일 경로 및 불러오기
    '''
    dir_name = "D:/240331-0dbm-rawdata/results/"#/test01"#dynamic-range-240329/"
    file_dir = (dir_name)
    file_name = "RMSE_DOS-data.csv"
    data_path = os.path.join(file_dir, file_name)
    target_data = pd.read_csv(data_path)
    n = 441

    '''
    클래스 인스턴스 생성
    '''
    save_path = dir_name
    slicer = DataFrameSlicer(target_data, slice_size=n)
    slicer.add_new_column(case='dos', max_point=10, step=1, number_interval=21)
    slicer.slice_dataframe()
    slicer.save_slices_to_csv(save_path, "0dBm")