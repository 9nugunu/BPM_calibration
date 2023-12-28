import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

def PlotSettings():
    plt.rcParams['axes.labelsize'] = 16
    # plt.rcParams["axes.labelweight"] = "bold"
    # plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 16
    plt.rcParams['image.cmap'] = 'jet'
    plt.rcParams['axes.titlesize'] = 16
    # plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["font.family"] = 'Times New Roman'
    plt.rcParams['lines.markersize'] ** 2

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

def BPM_curve_fit(x, target, fit_num):
    # print(FormOfpoly)
    if fit_num == 1:
        popt, pcov = curve_fit(fit_1st, x, target)
    elif fit_num == 3:
        popt, pcov = curve_fit(fit_3rd, x, target)
    elif fit_num == 5:
        popt, pcov = curve_fit(fit_5th, x, target)
    elif fit_num == '2D-3rd':
        popt, pcov = curve_fit(fit_2D, x, target)        
    return popt

def fit_1st(x, a, b): # , c, d, e, f, e, f, g, h, i, j
    return a*x + b
    # return a*x**9 + b*x**8 + c*x**7 + d*x**6 + e*x**5 + f*x**4 + g*x**3 + h*x**2 + i*x + j
    # return a*x**7 + b*x**6 + c*x**5 + d*x**4 + e*x**3 + f*x**2 + g*x + h
    # return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f
    # return a*x**3 + b*x**2 + c*x + d

def fit_3rd(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d
    # return a*x**3 + c*x + d

def fit_5th(x, a, b, c, d, e, f):
    # return a*x**9 + b*x**8 + c*x**7 + d*x**6 + e*x**5 + f*x**4 + g*x**3 + h*x**2 + i*x + j
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f

def fit_2D(xy, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
    x, y = xy
    # return a*x**9 + b*x**8 + c*x**7 + d*x**6 + e*x**5 + f*x**4 + g*x**3 + h*x**2 + i*x + j
    return a*x**3*y**3+b*x**2*y**3+c*x*y**3+d*y**3+e*x**3*y**2+f*x**2*y**2+g*x*y**1+h*y**1+i*x**3*y**1+j*x**2*y**1+k*x*y**1+l*y**1+m*x**3+n*x**2+o*x+p

def rms(series):
    print(series)
    return np.sqrt(np.mean(series**2))

def optimized_func(raw_data_, Wanted_data, cal_range_, fit_num):
    ver = 0
    # print(raw_data_.head())
    # print(raw_data_.groupby('x').mean())

    if ver == 0:
        mean_x = raw_data_.groupby('x').mean()
        mean_y = raw_data_.groupby('y').mean()

        xdata_fit = mean_x[abs(mean_x.index) <= cal_range_][Wanted_data['X']]
        ydata_fit = mean_y[abs(mean_y.index) <= cal_range_][Wanted_data['Y']]

    elif ver == 1:
        filtered_data = raw_data_[raw_data_['x'] == raw_data_['y']]
        xdata_fit = filtered_data[abs(filtered_data['x']) <= cal_range_][Wanted_data['X']]
        ydata_fit = filtered_data[abs(filtered_data['y']) <= cal_range_][Wanted_data['Y']]
        xdata_fit.index = filtered_data[abs(filtered_data['x']) <= cal_range_]['x']
        ydata_fit.index = filtered_data[abs(filtered_data['y']) <= cal_range_]['y']

    elif ver == 2:
        filtered_data = raw_data_
        xdata_fit = filtered_data[abs(filtered_data['x']) <= cal_range_][Wanted_data['X']]
        ydata_fit = filtered_data[abs(filtered_data['y']) <= cal_range_][Wanted_data['Y']]
        xdata_fit.index = filtered_data[abs(filtered_data['x']) <= cal_range_]['x']
        ydata_fit.index = filtered_data[abs(filtered_data['y']) <= cal_range_]['y']

    elif ver == 3:
        mean_x = raw_data_.groupby('x').agg(np.median)
        mean_y = raw_data_.groupby('y').agg(np.median)

        xdata_fit = mean_x[abs(mean_x.index) <= cal_range_][Wanted_data['X']]
        ydata_fit = mean_y[abs(mean_y.index) <= cal_range_][Wanted_data['Y']]

    #raw_data_.groupby('x').mean()
    #raw_data_.groupby('y').mean()

    # xdata_fit = mean_x[abs(mean_x.index) <= cal_range_][Wanted_data['X']]
    # ydata_fit = mean_y[abs(mean_y.index) <= cal_range_][Wanted_data['Y']]

    if fit_num != "2D-3rd":
        poptx = BPM_curve_fit(xdata_fit.values, xdata_fit.index, fit_num)
        popty = BPM_curve_fit(ydata_fit.values, ydata_fit.index, fit_num)
    else:
        poptx = BPM_curve_fit((xdata_fit.values, ydata_fit.values), xdata_fit.index, fit_num)
        popty = BPM_curve_fit((xdata_fit.values, ydata_fit.values), ydata_fit.index, fit_num)

    if fit_num == 1:
        cal_x_ = fit_1st(np.array(raw_data_[Wanted_data['X']]), *poptx)
        cal_y_ = fit_1st(np.array(raw_data_[Wanted_data['Y']]), *popty)
    elif fit_num == 3:
        cal_x_ = fit_3rd(np.array(raw_data_[Wanted_data['X']]), *poptx)
        cal_y_ = fit_3rd(np.array(raw_data_[Wanted_data['Y']]), *popty)
    elif fit_num == 5:
        cal_x_ = fit_5th(np.array(raw_data_[Wanted_data['X']]), *poptx)
        cal_y_ = fit_5th(np.array(raw_data_[Wanted_data['Y']]), *popty)
    elif fit_num == '2D-3rd':
        # print(Wanted_data['X'])
        # xy_values = raw_data_[Wanted_data['X', 'Y']]
        x_2dset = np.array(raw_data_[Wanted_data['X']])
        y_2dset = np.array(raw_data_[Wanted_data['Y']])
        dataset = np.array(x_2dset), np.array(y_2dset)
        #print(dataset)
        cal_x_ = fit_2D(dataset, *poptx)
        cal_y_ = fit_2D(dataset, *popty)

    return cal_x_, cal_y_

def ErrorWrtRange(data_, Wanted_data_, cal_range_, step_, error_dict_, errors_all_, cal_method_):
    # data_.drop([' Time', ' Type', ' 1Ch', ' 2Ch',  ' 3Ch', ' 4Ch'], axis=1, inplace=True)

    # error_dict = {}
    range_values = np.arange(step_, cal_range_+step_, step_)
    # errors_all = {1: [], 3: [], 5: [], '2D-3rd': []}
    for fit in cal_method_:
        fit_num = fit
        # print("="*300)
        cal_x_, cal_y_ = optimized_func(data_, Wanted_data_, cal_range_, fit_num)
        # print("*"*300)
        data_['cal_X'], data_['cal_Y'] = cal_x_, cal_y_

        cal_offset = data_[(data_['x'] == 0) & (data_['y'] == 0)][['cal_X', 'cal_Y']]

        data_['cal_X'], data_['cal_Y'] = data_['cal_X'] - cal_offset['cal_X'].values, data_['cal_Y'] - cal_offset['cal_Y'].values

        error_list = []
        for value in range_values:
            mask = (np.abs(data_['x']) <= value) & (np.abs(data_['y']) <= value)
            # mask_y = (np.abs(Wanted_data['y']) <= value)
    
            # mask = np.abs(Wanted_data) <= value
            filtered_data = data_[mask]
            filtered_Wanted_data_x = filtered_data['x']
            filtered_cal_x = filtered_data['cal_X']
            # print(filtered_cal_x)
            
            # masky = ((np.abs(Wanted_data['x']) <= value))
            # filtered_Wanted_data_y = data_['y'][mask]
            # filtered_cal_y = data_['cal_Y'][mask]
            filtered_Wanted_data_y = filtered_data['y']
            filtered_cal_y = filtered_data['cal_Y']

            error_x = filtered_Wanted_data_x - filtered_cal_x
            error_y = filtered_Wanted_data_y - filtered_cal_y
            error_z = np.mean(np.sqrt(error_x**2 + error_y**2))
            # print(error_z)
            error_list.append(error_z*10**3)
            # plt.scatter(filtered_cal_x, filtered_cal_y, label=f'n = {fit}')
            # plt.show()
            # if fit not in all_errors:
            #     all_errors[fit] = []
            # print(error_list)
        # print(f"fit: {fit}, 5th_error_list: {error_list[5]} **************************************")
        errors_all_[fit].append(error_list)
            
        error_dict_[fit] = error_list
        
    return error_dict_, errors_all_
