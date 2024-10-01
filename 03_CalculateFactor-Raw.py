import os, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import TestBench_data_processing as tb_dataprocessing
from scipy.optimize import curve_fit

tb_dataprocessing.PlotSettings()

def load_coeffs_from_csv(file_path):
    coeffs_df = pd.read_csv(file_path)
    # print(file_path.split('$')[1])
    # print(coeffs_df)
    # os._exit(1)
    # coeffs_df.dropna(inplace=True)
    file_key = file_path.split('$')[1]

    coeffs_df.index = [f"{file_key}_fit_num_{row_name}" for row_name in coeffs_df['fit_num']]
    coeffs_df.drop('fit_num', axis=1, inplace=True)
    coeffs_dict = coeffs_df.T.to_dict('list')
    print(coeffs_dict)
    return coeffs_dict

def apply_optimization(raw_data, coeffs_dict, fit_num):
    coeffs = coeffs_dict[fit_num]
    dos_method = fit_num.split('_fit_num_')[0]
    fit_num_end = fit_num[-1]
    # print(coeffs)
    fitting_function = None

    if fit_num_end == '1':
        fitting_function = lambda x, a, b: a * x + b# + offset
    elif fit_num_end == '3':
        fitting_function = lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d# + offset
    elif fit_num_end == '5':
        fitting_function = lambda x, a, b, c, d, e, f: a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f# + offset
    elif fit_num_end == "d":
        fitting_function = lambda xy, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p: (
            a * xy[0]**3 * xy[1]**3 + b * xy[0]**2 * xy[1]**3 + c * xy[0] * xy[1]**3 + d * xy[1]**3 +
            e * xy[0]**3 * xy[1]**2 + f * xy[0]**2 * xy[1]**2 + g * xy[0] * xy[1]**2 + h * xy[1]**2 +
            i * xy[0]**3 * xy[1]**1 + j * xy[0]**2 * xy[1]**1 + k * xy[0] * xy[1]**1 + l * xy[1]**1 + 
            m * xy[0]**3 + n * xy[0]**2 + o * xy[0] + p# + offset
        )
    
    if fit_num_end == '1':
        print("=="*100)
        print("linear calibration")
        print(coeffs[2:4])
        raw_data[f'{dos_method}_X(A)_{fit_num_end}'] = fitting_function(raw_data[' X(A)'], *coeffs[:2]) # , *coeffs[2:3]
        raw_data[f'{dos_method}_Y(A)_{fit_num_end}'] = fitting_function(raw_data[' Y(A)'], *coeffs[2:4]) # , *coeffs[3:5], *coeffs[5:6]
    elif fit_num_end == '3':
        print("=="*100)
        print("3rd polynomial fit calibration")
        print(coeffs[4:8])
        raw_data[f'{dos_method}_X(A)_{fit_num_end}'] = fitting_function(raw_data[' X(A)'], *coeffs[:4]) # , *coeffs[4:5]
        raw_data[f'{dos_method}_Y(A)_{fit_num_end}'] = fitting_function(raw_data[' Y(A)'], *coeffs[4:8]) # , *coeffs[9:10]

    elif fit_num_end == '5':
        print("=="*100)
        print("5th polynomial fit calibration")
        # print(coeffs)
        # print(len(coeffs))
        # print(coeffs[:6])
        raw_data[f'{dos_method}_X(A)_{fit_num_end}'] = fitting_function(raw_data[' X(A)'], *coeffs[:6])
        raw_data[f'{dos_method}_Y(A)_{fit_num_end}'] = fitting_function(raw_data[' Y(A)'], *coeffs[6:12])
    elif fit_num_end == 'd':
        print("=="*100)
        print("2D-3rd polynomial fit calibration")
        # print(coeffs)
        xy_data = [list(xy) for xy in zip(raw_data[' X(A)'], raw_data[' Y(A)'])]
        x_2dset = np.array(raw_data[Wanted_data["X"]])
        y_2dset = np.array(raw_data[Wanted_data["Y"]])
        dataset = np.array(x_2dset), np.array(y_2dset)
        x, y = dataset
        # print(xy_data[0])
        print(*coeffs[17:33])
        # os._exit(1)
        raw_data[f'{dos_method}_X(A)_{fit_num_end}'] = [fitting_function(xy, *coeffs[:16]) for xy in xy_data] # tb_dataprocessing.fit_2D(dataset, *coeffs[:16])
        raw_data[f'{dos_method}_Y(A)_{fit_num_end}'] = [fitting_function(xy, *coeffs[16:33]) for xy in xy_data] #tb_dataprocessing.fit_2D(dataset, *coeffs[17:33])
    else:
        raw_data = "No fit cases"
    
    print(coeffs_dict[fit_num])
    return raw_data[[f'{dos_method}_X(A)_{fit_num_end}', f'{dos_method}_Y(A)_{fit_num_end}']]

'''
변수 지정
계수값 csv파일 경로로 옮겨놔야함.

'''
number_interval = 21
step = 1
max_point = 10
cal_range = 6
Wanted_data = {"X": " X(A)", "Y": " Y(A)"}

optimizer = tb_dataprocessing.Optimizer()

# file_dir = ("../-5_5_dataset/" + '2port/' + f"BPM01_352MHz_variAmp/")  # + filename # 'PAPER_ONLY_0825/' +
file_dir = "C:/Users/9nugu/Documents/03-04-combined/results/"
samples = "5000 samples/"
file_sample_dir = file_dir + samples
output_dir = os.path.join(file_sample_dir, "Calcul_single_results")
file_list = [file for file in os.listdir(file_sample_dir) if file.lower().endswith('.csv')]
print(file_list)

coeffs_file_dir= file_sample_dir + "coefficient/"
coeffs_files = [file for file in os.listdir(coeffs_file_dir) if file.lower().endswith('.csv')]

coeffs_dict = {}

for coeffs_file in coeffs_files:
    coeffs_file_path = os.path.join(coeffs_file_dir, coeffs_file)
    coeffs_dict.update(load_coeffs_from_csv(coeffs_file_path))
# print(type(coeffs_dict))

# print(coeffs_dict)
# print(coeffs_dict.keys())

for index, i in enumerate(file_list):
    data_path = os.path.join(file_sample_dir, i)
    raw_data = pd.read_csv(data_path, index_col=False)

    if 'x' not in raw_data.columns and 'y' not in raw_data.columns:
        raw_data["x"], raw_data["y"] = tb_dataprocessing.add_col_axis(
            number_interval, step, max_point
        )

    selected_data = raw_data[[' X(A)', ' Y(A)', "x", "y"]]
    calibrated_single_data = pd.DataFrame(index=selected_data.index)
    optimized_data = selected_data.copy() #pd.DataFrame(index=selected_data.index)

    for fit_ver in coeffs_dict.keys():
        dos_method = fit_ver.split('_fit_num_')[0]
        fit_num = fit_ver.split('_fit_num_')[1]
        # os._exit(1)
        cols = [f'{dos_method}_X(A)_{fit_num}', f'{dos_method}_Y(A)_{fit_num}']
        optimized_data[cols] = apply_optimization(selected_data.copy(), coeffs_dict, fit_ver)
        print(fit_ver)        
    # print(optimized_data)
    # print("***"*100)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.chdir(output_dir)
    optimized_data.to_csv(f'{index // 10}{index % 10}_Calibrated-single-raw-data.csv', index=False)
    os.chdir("../../..")

# x_dummy = [np.arange(-max_point, max_point + step, step)] * number_interval
# y_dummy = [i for i in np.arange(-max_point, max_point + step, step) for _ in range(number_interval)]
# plt.scatter(x_dummy, y_dummy, s=40, marker=".", edgecolor="b")
# plt.scatter(
#     optimized_data["DOS_{all}_X(A)_d"],
#     optimized_data["DOS_{all}_Y(A)_d"],
#     s=30,
#     marker="o",
#     facecolor="none",
#     edgecolors="r",
# )
# # plt.title("Linear calibration result")
# plt.xlabel("X [mm]")
# plt.ylabel("Y [mm]")
# plt.xlim([-max_point - step, max_point + step])
# plt.ylim([-max_point - step, max_point + step])
# plt.gca().set_aspect("equal")
# plt.tight_layout()
# # plt.ylabel("K$_{x, y}$ X DOS ($\Delta/\Sigma$)")
# plt.grid()
# plt.show()
        # selected_data.loc[:, cols] = optimized_data[cols].copy()
# print(selected_data)
    # plt.plot[(raw_data["x"] ==  0) & (raw_data["y"] == 0)][["cal_X", "cal_Y"]]