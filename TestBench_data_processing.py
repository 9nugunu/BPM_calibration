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

def add_col_axis(raw_data, step, measure_range):
    if step == 0.25 or step == 0.5:
        number_interval = 21
        x = list(np.arange(5, -5.5, -0.5))
        repeated_x = x * number_interval
        # print(repeated_x)
        y = [i for i in np.arange(5, -5.5, -0.5) for _ in range(number_interval)]
        # row_data['x'], row_data['y'] = repeated_x, y
    else:
        number_interval = 21
        x = [i for i in range(measure_range, -measure_range-1, -1)]
        repeated_x = x * number_interval
        # print(repeated_x)
        y = [i for i in range(measure_range, -measure_range-1, -1) for _ in range(number_interval)]
        # row_data['x'], row_data['y'] = repeated_x, y
    return number_interval, repeated_x, y