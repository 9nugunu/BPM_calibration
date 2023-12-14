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

def BPM_curve_fit(x, y):
    # print(FormOfpoly) 
    popt, pcov = curve_fit(FormOfpoly, x, y)
    return popt

def FormOfpoly(x, a, b, c, d, e, f, g, h): # , c, d, e, f, e, f, g, h, i, j
    if fitting_number == 3:
        return a*x**3 + b*x**2 + c*x + d
        # return a*x**9 + b*x**8 + c*x**7 + d*x**6 + e*x**5 + f*x**4 + g*x**3 + h*x**2 + i*x + j
    elif fitting_number == 1:
        # print("Fome test", x)
        return a*x + b
    elif fitting_number == 5:
        return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f    
    elif fitting_number == 7:
        return a*x**7 + b*x**6 + c*x**5 + d*x**4 + e*x**3 + f*x**2 + g*x + h
    # return a*x**9 + b*x**8 + c*x**7 + d*x**6 + e*x**5 + f*x**4 + g*x**3 + h*x**2 + i*x + j
    # return a*x**7 + b*x**6 + c*x**5 + d*x**4 + e*x**3 + f*x**2 + g*x + h
    # return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f
    # return a*x**3 + b*x**2 + c*x + d

def optimized_func(data_, params_, max_point, Wanted_data):
    # data_, params_ = data, params
    fitting_range = 5
    filtered_for_sensitivity = data_[data_['x'] == data_['y']]
    
    xdata_fit = filtered_for_sensitivity[abs(filtered_for_sensitivity['x']) <= max_point][Wanted_data]
    ydata_fit = filtered_for_sensitivity[abs(filtered_for_sensitivity['y']) <= max_point][Wanted_data_y]
    params_fit_x = filtered_for_sensitivity[(abs(filtered_for_sensitivity['x']) <= max_point)]['x']
    params_fit_y = filtered_for_sensitivity[(abs(filtered_for_sensitivity['y']) <= max_point)]['y']
    poptx = BPM_curve_fit(xdata_fit, params_fit_x)
    popty = BPM_curve_fit(ydata_fit, params_fit_y)
    
    cal_x_ = FormOfpoly(np.array(data_[Wanted_data_x]), *poptx)
    # print("fitting factors: ", *poptx)
    cal_y_ = FormOfpoly(np.array(data_[Wanted_data_y]), *popty)
    return cal_x_, cal_y_