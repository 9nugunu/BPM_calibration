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

def BPM_curve_fit(x, y, fit_num):
    # print(FormOfpoly)
    if fit_num == 1:
        popt, pcov = curve_fit(fit_1st, x, y)
    elif fit_num == 3:
        popt, pcov = curve_fit(fit_3rd, x, y)
    elif fit_num == 5:
        popt, pcov = curve_fit(fit_5th, x, y)
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

def optimized_func(raw_data_, Wanted_data, cal_range_, fit_num):
    # print(raw_data_.head())
    # print(raw_data_.groupby('x').mean())

    mean_x = raw_data_.groupby('x').mean()
    mean_y = raw_data_.groupby('y').mean()

    xdata_fit = mean_x[abs(mean_x.index) <= cal_range_][Wanted_data['X']]
    ydata_fit = mean_y[abs(mean_y.index) <= cal_range_][Wanted_data['Y']]

    poptx = BPM_curve_fit(xdata_fit.values, xdata_fit.index, fit_num)
    popty = BPM_curve_fit(ydata_fit.values, ydata_fit.index, fit_num)
    
    # print("*"*100)
    # print(len(poptx), popty)
    # print("*"*100)

    if fit_num == 1:
        cal_x_ = fit_1st(np.array(raw_data_[Wanted_data['X']]), *poptx)
        cal_y_ = fit_1st(np.array(raw_data_[Wanted_data['Y']]), *popty)
    elif fit_num == 3:
        cal_x_ = fit_3rd(np.array(raw_data_[Wanted_data['X']]), *poptx)
        cal_y_ = fit_3rd(np.array(raw_data_[Wanted_data['Y']]), *popty)
    elif fit_num == 5:
        cal_x_ = fit_5th(np.array(raw_data_[Wanted_data['X']]), *poptx)
        cal_y_ = fit_5th(np.array(raw_data_[Wanted_data['Y']]), *popty)

    # cal_x_, cal_y_ = 0, 0

    # print(cal_x_, cal_y_)
    return cal_x_, cal_y_

def ErrorWrtRange(data_, Wanted_data_, max_point_, cal_range_, step_):
    data_.drop([' Time', ' Type', ' 1Ch', ' 2Ch',  ' 3Ch', ' 4Ch', ' X(B)', ' Y(B)'], axis=1, inplace=True)

    error_dict = {}
    range_values = np.arange(step_, cal_range_+step_, step_)
    errors_all = {1: [], 3: [], 5: [], 7: [], 9: []}
    for fit in [1, 3, 5]:
        fit_num = fit
        print("="*300)
        cal_x_, cal_y_ = optimized_func(data_, Wanted_data_, cal_range_, fit_num)
        print("*"*300)
        data_['cal_X'], data_['cal_Y'] = cal_x_, cal_y_
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
        errors_all[fit].append(error_list)
            
        error_dict[fit] = error_list
        
        # errors_std, errors_se, errors_mean, errors_rms = {}, {}, {}, {}
        # sample_size = len(next(iter(errors_all.values())))  # 가정: 모든 fit 값들에 대해 샘플의 크기가 동일하다.
        # for fit, errors in errors_all.items():
        #     print(len(errors))
        #     errors_std[fit] = np.std(errors, axis=0)
        #     errors_se[fit] = errors_std[fit] / np.sqrt(sample_size)
        #     errors_mean[fit] = np.mean(errors, axis=0)
        #     errors_rms[fit] = np.sqrt(np.mean(np.array(errors)**2, axis=0))

    # Plot the errors
    plt.figure()
    markers = ['^', 's', 'D', '.', '<']
    p_color = ['r', 'b', 'magenta', 'g', 'grey']
    for i, (fit, error_list) in enumerate(error_dict.items()):
        plt.plot(range_values, error_list, label=f'n = {fit}', marker=markers[2-i], c=p_color[i])

        # Find the maximum x where error is less than or equal to 0.10
        # max_x = np.max(np.array(range_values)[np.array(error_list) <= 0.10])
        # if i < 2:
        #     y_value_at_3 = error_list[np.where(np.array(range_values) == 3.0)[0][0]] # Extracting the y-value at x=3
        #     plt.annotate(f"{y_value_at_3:.2f}", (3, y_value_at_3), textcoords="offset points", xytext=(-2,-40), color=p_color[i], ha='right', arrowprops=dict(arrowstyle="->", color=p_color[i]))
        
        #     # Annotation for x=4
        #     y_value_at_3_5 = error_list[np.where(np.array(range_values) == 3.5)[0][0]] # Extracting the y-value at x=4
        #     plt.annotate(f"{y_value_at_3_5:.2f}", (3.5, y_value_at_3_5), textcoords="offset points", xytext=(14,10), color=p_color[i], ha='right')
            
        #     # Annotation for x=4
        #     y_value_at_4 = error_list[np.where(np.array(range_values) == 4.0)[0][0]] # Extracting the y-value at x=4
        #     plt.annotate(f"{y_value_at_4:.2f}", (4, y_value_at_4), textcoords="offset points", xytext=(80,-23), color=p_color[i], ha='right', arrowprops=dict(arrowstyle="->", color=p_color[i]))
        
        plt.axvline(3.0, color='gray', linestyle='--')
        plt.axvline(4.0, color='gray', linestyle='--')
    plt.axhline(100, color='gray', linestyle='--')
    plt.xlabel('wire movement range x, y [mm]')
    plt.ylabel(u'Average error [\u03bcm]')
    # plt.ylabel(u"\u03bcs")
    plt.xticks(range_values)
    # y_ticks = np.arange(0, 201, 25)
    # plt.yticks(y_ticks)
    # plt.ylim(0, 0.3)
    plt.legend()
    plt.show()