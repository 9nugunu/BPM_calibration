import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

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

def BPM_curve_fit(x, y):
    # print(FormOfpoly) 
    popt, pcov = curve_fit(FormOfpoly, x, y)
    return popt

def FormOfpoly(x, a, b, c, d, e, f, g, h): # , c, d, e, f, e, f, g, h, i, j
    fit_num = 1
    if fit_num == 3:
        return a*x**3 + b*x**2 + c*x + d
        # return a*x**9 + b*x**8 + c*x**7 + d*x**6 + e*x**5 + f*x**4 + g*x**3 + h*x**2 + i*x + j
    elif fit_num == 1:
        # print("Fome test", x)
        return a*x + b
    elif fit_num == 5:
        return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f    
    elif fit_num == 7:
        return a*x**7 + b*x**6 + c*x**5 + d*x**4 + e*x**3 + f*x**2 + g*x + h
    # return a*x**9 + b*x**8 + c*x**7 + d*x**6 + e*x**5 + f*x**4 + g*x**3 + h*x**2 + i*x + j
    # return a*x**7 + b*x**6 + c*x**5 + d*x**4 + e*x**3 + f*x**2 + g*x + h
    # return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f
    # return a*x**3 + b*x**2 + c*x + d

def optimized_func(data_, Wanted_data, max_point):

    mean_x = data_.groupby('x').mean()
    mean_y = data_.groupby('y').mean()

    xdata_fit = mean_x[abs(mean_x.index) <= max_point][Wanted_data['X']]
    ydata_fit = mean_y[abs(mean_y.index) <= max_point][Wanted_data['Y']]

    params_fit_x = mean_x[abs(mean_x.index) <= max_point][Wanted_data['X']].index#data_[abs(data_[Wanted_data['X']]) <= max_point]['x']
    params_fit_y = mean_y[abs(mean_y.index) <= max_point][Wanted_data['Y']].index#data_[abs(data_[Wanted_data['Y']]) <= max_point]['y']

    poptx = BPM_curve_fit(xdata_fit.values, xdata_fit.index)
    popty = BPM_curve_fit(ydata_fit.values, ydata_fit.index)
    
    # cal_x_ = 0
    # cal_y_ = 0
    cal_x_ = FormOfpoly(np.array(data_[Wanted_data['X']]), *poptx)
    # print("fitting factors: ", *poptx)
    cal_y_ = FormOfpoly(np.array(data_[Wanted_data['Y']]), *popty)

    print(cal_x_, cal_y_)
    return cal_x_, cal_y_