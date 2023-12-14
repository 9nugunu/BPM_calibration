import matplotlib.pyplot as plt
import numpy as np

def PlotSettings():
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 12
    plt.rcParams['image.cmap'] = 'jet'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["font.family"] = 'Times New Roman'

def add_col_axis(number_interval, step, max_point):
    '''
    측정 데이터에 x, y original 값 추가하는 함수
    모터기준 -point ~ +point 이동 기준
    반대방향으로 이동 시 할당값 변경 필요
    '''
    x = [i for i in np.arange(max_point, -max_point-step, -step)]
    repeated_x = x * number_interval
    # print(len(repeated_x))
    y = [i for i in x for _ in range(number_interval)]
    # row_data['x'], row_data['y'] = repeated_x, y

    return repeated_x, y