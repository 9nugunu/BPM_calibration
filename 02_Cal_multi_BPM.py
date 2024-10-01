import os, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import TestBench_data_processing as tb_dataprocessing
import matplotlib.ticker as ticker
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable

'''
작성일: 2024. 01
작성자: 김근우
코드목적: n개의 dos file -> averaged 후 보정 진행 
'''
tb_dataprocessing.PlotSettings()

number_interval = 21

step = 1
max_point = 10
cal_range = 8


Port = "2port/"
Wanted_data = {"X": " X(A)", "Y": " Y(A)"}
target_freq = "352"
optimizer = tb_dataprocessing.Optimizer()

#r"$S_{\bar{x}}$": 0, 
sensitivities = {r"$S_{diag}$": 1, r"$S_{axis}$": 4, r"$S_{all}$": 2}
DOS_selection = {r"$S_{diag}$":r"$DOS_{diag}$", r"$S_{axis}$":r"$DOS_{axis}$", r"$S_{all}$":r"$DOS_{all}$"}

# r"$\mathregular{DOS_{all}}$"
# sensitivities = {r"$S_{onaxis}$": 4}
for sensitivity, fit_ver in sensitivities.items():
    current_DOS = DOS_selection[sensitivity]
    coeffs_data = []

    print(f"Current DOS method: {current_DOS}")
    data = pd.DataFrame()
    optimizer.set_ver(fit_ver)
    sensi_str = sensitivity.strip("$")
    if sensitivity == r"$S_{\bar{x}}$":
        sensi_str = "S_bar_x"

    if sensitivity == r"$S_{all}$":
        cal_method = [1, 3, 5, "2D-3rd"]  # [1, 3, 5, '2D-3rd']
    else:
        cal_method = [1, 3, 5]  # [1, 3, 5, '2D-3rd']

    # filename = 'cal_paper__' + '1' + '_4port_01_0.25.csv'
    filename = "BPM01_352MHz_8dBm_2port_01_-10to10_100_20240109_201518.csv"
    # file_dir = ("../-5_5_dataset/" + Port + f"BPM01_{target_freq}MHz_variAmp/")  # + filename # 'PAPER_ONLY_0825/' +
    file_dir = "C:\\Users\\9nugu\\Documents\\03-04-combined\\results\\5000 samples"
    file_list = os.listdir(file_dir)
    file_list = [file for file in file_list if file.lower().endswith('.csv')]
    # file_list = os.listdir(csv_files)
    print(type(file_list))
    # os.chdir('../' + file_dir)
    print(os.getcwd())
    print(file_list)
    # time.sleep(3)

    # cal_offset = data[(data["x"] == 0) & (data["y"] == 0)][["cal_X", "cal_Y"]]
    """
    n개 데이터 평균 → 하나의 데이터 프레임
    """
    data_sum = pd.DataFrame()
    data_squared_sum = pd.DataFrame()

    for i in file_list:
        data_path = os.path.join(file_dir, i)
        raw_data = pd.read_csv(data_path, index_col=False)
        # raw_data.drop(
        #     [" 1Ch", " 2Ch", " 3Ch", " 4Ch"], axis=1, inplace=True
        # )
        raw_data["x"], raw_data["y"] = tb_dataprocessing.add_col_axis(
            number_interval, step, max_point
        )
        selected_data = raw_data#[[Wanted_data["X"], Wanted_data["Y"], "x", "y"]]
        # squared_data = selected_data ** 2
        # plt.plot[(raw_data["x"] == 0) & (raw_data["y"] == 0)][["cal_X", "cal_Y"]]
        if data_sum.empty:
            data_sum = selected_data
            # data_squared_sum = squared_data

        else:
            data_sum += selected_data
            # data_squared_sum += squared_data
            # data += raw_data[[Wanted_data["X"], Wanted_data["Y"], "x", "y"]]

    data = data_sum / len(file_list)
    # data_squared_mean = data_squared_sum / len(file_list)
    # data_std_dev = np.sqrt(data_squared_mean - data ** 2)
    # data_std_dev.to_csv("data_std_dev.csv", index=False)
    data.to_csv("data_averaged.csv", index=False)
    data_origin = data

    # plt.figure(1)
    # plt.scatter(data['x'], data_std_dev[' X(A)'])
    # plt.scatter(data['y'], data_std_dev[' Y(A)'])

    # plt.figure(2)
    # plt.scatter(data_std_dev[' X(A)'], data_std_dev[' Y(A)'])
    # plt.grid()
    # plt.show()
    # os._exit(1)
    '''
    Measured DOS data plotting
    '''
    if fit_ver == 1:
        filtered_data = data[data["x"] <= 8]
        plt.figure(1, figsize=(6,6))
        
        plt.grid()
        plt.scatter(data[(data["x"] == data["y"]) & (np.abs(data['x']) <= cal_range)][Wanted_data["X"]], data[(data["x"] == data["y"]) & (np.abs(data['x']) <= cal_range)][Wanted_data["Y"]], s=40, marker="^", edgecolor="r", fc='r', label=r"$\mathregular{DOS_{diag}}$")
        plt.scatter(data[(data["x"] == 0) & (np.abs(data['y']) <= cal_range)][Wanted_data["X"]], data[(data["x"] == 0) & (np.abs(data['y']) <= cal_range)][Wanted_data["Y"]], s=30, marker="D", edgecolor="limegreen", fc='limegreen', label=r"$\mathregular{DOS_{axis}}$")
        plt.scatter(data[(data["y"] == 0) & (np.abs(data['x']) <= cal_range)][Wanted_data["X"]], data[(data["y"] == 0) & (np.abs(data['x']) <= cal_range)][Wanted_data["Y"]], s=30, marker="D", edgecolor="limegreen", fc='limegreen')
        plt.scatter(data[(np.abs(data['x']) <= cal_range) & (np.abs(data['y']) <= cal_range)][Wanted_data["X"]], data[(np.abs(data['x']) <= cal_range) & (np.abs(data['y']) <= cal_range)][Wanted_data["Y"]], s=40, marker=".", edgecolor="b", fc='b', label=r"$\mathregular{DOS_{all}}$")
        # plt.yticks([0.4, 0.3, 0.2, 0.1, 0.0, -0.1])
        plt.legend(fontsize=16, framealpha=0.95, loc='upper left')
        plt.title("Measured DOS data", fontweight='bold', fontsize=22)

        plt.xticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])
        plt.yticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])
        plt.xlabel("X DOS data")
        plt.ylabel("Y DOS data")
        # print(os.getcwd())
        plt.savefig('RAW_DOS.png',
            format='png',
            dpi=400,
        bbox_inches='tight')
        plt.close()

        # DOS_offset = data[(data["x"] == 0) & (data["y"] == 0)][Wanted_data["X"], Wanted_data["Y"]]
        cal_offset = data[(data["x"] == 0) & (data["y"] == 0)][
            [Wanted_data["X"], Wanted_data["Y"]]
        ]
        x_offset = round(cal_offset[Wanted_data["X"]].values[0], 3)
        y_offset = round(cal_offset[Wanted_data["Y"]].values[0], 3)
        print(rf"x_offset: {x_offset}")
        print(f"y_offset: {y_offset}")
        os._exit(1)
        # median_x = data.groupby("x").agg(np.median)
        # median_y = data.groupby("y").agg(np.median)
        # print(mean_same_x)
        '''
        X dos data plotting
        '''
        filtered_data = data[abs(data["x"]) <= 8]

        fig3, ax3 = plt.subplots()
        ax3.set_title("Three selected X DOS shapes", fontsize=22, fontweight='bold')

        # 메인 플롯
        ax3.plot(
            filtered_data[filtered_data["x"] == filtered_data["y"]]["x"],
            filtered_data[filtered_data["x"] == filtered_data["y"]][Wanted_data["X"]],
            label=r"$X_{diag}$",
            c="b",
        )
        ax3.plot(
            filtered_data[filtered_data["y"] == 0]["x"],
            filtered_data[filtered_data["y"] == 0][Wanted_data["X"]],
            label=r"$X_{axis}$",
            c="r",
            linestyle="--",
        )
        ax3.scatter(filtered_data["x"], filtered_data[Wanted_data["X"]], label=r"$X_{all}$", s=8, c="black")
        ax3.grid()

        # 확대된 인셋 플롯
        axins = zoomed_inset_axes(ax3, 2.5, loc="lower right", axes_kwargs={"fc": "lightgray"})
        axins.plot(
            filtered_data[filtered_data["x"] == filtered_data["y"]]["x"],
            filtered_data[filtered_data["x"] == filtered_data["y"]][Wanted_data["X"]],
            label=r"$DOS_{diag}$",
            c="b",
        )
        axins.plot(
            filtered_data[filtered_data["y"] == 0]["x"],
            filtered_data[filtered_data["y"] == 0][Wanted_data["X"]],
            label=r"$DOS_{axis}$",
            c="r",
            linestyle="--",
        )
        axins.scatter(
            filtered_data["x"], 
            filtered_data[Wanted_data["X"]], 
            label=r"$DOS_{all}$", 
            s=2, 
            c="black"
        )
        axins.set_xlim(5.40, 8.45)
        axins.set_ylim(0.49, 0.72)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.grid()

        mark_inset(ax3, axins, loc1=2, loc2=1, ec="0.5")
        ax3.set_xlabel("X position [mm]", fontsize=18)
        ax3.set_ylabel(r"$\Delta_{x}/\Sigma_{x}$", fontsize=18)
        ax3.set_xticks(range(-8, 9, 4))
        ax3.legend(fontsize=14)
        plt.savefig("Four_diff_sensiti_x.png", format="png", dpi=500, bbox_inches="tight")
        # plt.show()
        plt.close()

    # print(data.head())
    x_dummy = np.arange(-max_point, max_point + step, step)
    y_dummy = x_dummy
    x_dummy_df = pd.DataFrame(x_dummy)
    print(x_dummy_df)
    # print(mean_same_x)

    """
    선형피팅 Sensitivity 출력
    1D-2D Residual 그림
    Residuals
    """
    # fig1.suptitle(f"{sensitivity} case" + f" @ {target_freq} MHz", fontsize=16, y=0.92)
    if fit_ver != 2:
        plt.figure(2, figsize=(13, 5))
    else:
        plt.figure(2, figsize=(18, 5))
    for i, fit in enumerate(cal_method):
        plt.subplot(1, len(cal_method), i + 1)
        if fit_ver == 2:
            # plt.rcParams["axes.titlesize"] = 60
            # plt.rcParams["font.size"] = 20
            plt.suptitle(f"Selected DOS data calibration results for {current_DOS}" + f" at {target_freq} MHz", fontsize=30, y=0.92, fontweight='bold')
        else:
            plt.suptitle(f"Selected DOS data calibration results for {current_DOS}" + f" at {target_freq} MHz", fontsize=30, y=0.92, fontweight='bold')
        # plt.suptitle(f"Selected DOS data calibration results for {current_DOS}" + f" @ {target_freq} MHz", fontsize=22, y=0.92, fontweight='bold')
        plt.grid()
        if fit == 1:
            plt.title("Linear estimation", fontsize=18)
        elif fit == 3:
            plt.title("3rd-order polynomial", fontsize=18)
        elif fit == 5:
            plt.title("5th-order polynomial", fontsize=18)
        elif fit == "2D-3rd":
            plt.title("2D-3rd polynomial", fontsize=18)

        """ fitting start"""
        cal_x, cal_y = optimizer.optimized_func(data, Wanted_data, cal_range, fit)
        data["cal_X"], data["cal_Y"] = cal_x, cal_y

        # mean_same_x = data.groupby("x").mean()["x", "cal_X"]
        # mean_same_y = data.groupby("y").mean()["y","cal_Y"]

        # plt.scatter(diff_x_mean["x"], diff_x_mean["diff_x_mean"], lw=0.8, marker="o", label="X", c='b')
        # plt.scatter(diff_y_mean["y"], diff_y_mean["diff_y_mean"], lw=0.8, marker="^", label="Y", c='r')
        if fit_ver == 1:
            equal_data = data[(data['x'] == data['y']) & (np.abs(data['x']) <= cal_range)]
            print(equal_data)
            absx_equal, absy_equal = equal_data["x"], equal_data["y"]
            x_equal, y_equal = equal_data["cal_X"], equal_data["cal_Y"]

            x_residuals = x_equal - absx_equal
            y_residuals = y_equal - absy_equal
            mean_residual_x = np.mean(np.abs(x_residuals))
            mean_residual_y = np.mean(np.abs(y_residuals))

            # print(data[data['x'] == data['y']])
            # data[data['x'] == data['y']].to_csv(f'./equal_{fit}.csv')
            # plt.scatter(absx_equal, absx_equal - x_equal, lw=0.8, marker="o", label=f"X, MAE: {mean_residual_x*1e3:.2f} μm", c='b')
            # print(data[data['x'] == data['y']])
            plt.plot(absx_equal, absx_equal - x_equal, lw=1, c='b', marker="o", label=f"X, MAE: {mean_residual_x*1e3:.2f} μm")
            # plt.scatter(absy_equal, absy_equal - y_equal, lw=0.8, marker="^", label=f"Y, MAE: {mean_residual_y*1e3:.2f} μm", c='r')
            plt.plot(absy_equal, absy_equal - y_equal, lw=1, c='r', linestyle='--',marker="^", label=f"Y, MAE: {mean_residual_y*1e3:.2f} μm")
            plt.ylim([-1, 1])
            plt.ylabel("Residuals [mm]")
        elif fit_ver == 4:
            absx_axis, absy_axis = data[(data['y']==0) & (np.abs(data['x']) <= cal_range)]["x"], data[(data['x']==0) & (np.abs(data['y']) <= cal_range)]["y"] 
            x_axis, y_axis = data[(data['y']==0) & (np.abs(data['x']) <= cal_range)]["cal_X"], data[(data['x']==0) & (np.abs(data['y']) <= cal_range)]["cal_Y"]

            x_residuals = x_axis - absx_axis
            y_residuals = y_axis - absy_axis
            mean_residual_x = np.mean(np.abs(x_residuals))
            mean_residual_y = np.mean(np.abs(y_residuals))

            # print(data[data['y']==0])
            # data[data['y']==0].to_csv(f'./on_axis_{fit}.csv')
            # plt.scatter(absx_axis, absx_axis - x_axis, lw=0.8, marker="o", label=f"X, MAE: {mean_residual_x*1e3:.2f} μm", c='b')
            plt.plot(absx_axis, absx_axis - x_axis, lw=1, c='b', marker="o", label=f"X, MAE: {mean_residual_x*1e3:.2f} μm")
            # plt.scatter(absy_axis, absy_axis - y_axis, lw=0.8, marker="^", label=f"Y, MAE: {mean_residual_y*1e3:.2f} μm", c='r')
            plt.plot(absy_axis, absy_axis - y_axis, lw=1, c='r', linestyle='--', marker='^', label=f"Y, MAE: {mean_residual_y*1e3:.2f} μm")
            plt.ylim([-1, 1])
            plt.ylabel("Residuals [mm]")
        elif fit_ver == 2:
            mask_x = np.abs(data['x']) <= cal_range
            mask_y = np.abs(data['y']) <= cal_range
            diff_x_mean = (data[mask_x]["x"] - data[mask_x]["cal_X"]).groupby(data["x"]).mean().reset_index(name='diff_x_mean')
            diff_y_mean = (data[mask_y]["y"] - data[mask_y]["cal_Y"]).groupby(data["y"]).mean().reset_index(name='diff_y_mean')
            print(diff_x_mean)
            print(diff_y_mean)
            
            # mean_residual_x = np.mean(np.abs(data[mask_x]["x"] - data[mask_x]["cal_X"]))
            # mean_residual_y = np.mean(np.abs(data[mask_x]["y"] - data[mask_x]["cal_Y"]))

            mean_residual_x = np.mean(np.abs(diff_x_mean['diff_x_mean']))
            mean_residual_y = np.mean(np.abs(diff_y_mean['diff_y_mean']))
            print(mean_residual_x,mean_residual_y)
            # os._exit(1)
            # plt.scatter(diff_x_mean["x"], diff_x_mean["diff_x_mean"], lw=0.8, marker="o", label=f"X, MAE: {mean_residual_x*1e3:.2f} μm", c='b')
            plt.plot(diff_x_mean["x"], diff_x_mean["diff_x_mean"], lw=1, c='b', marker="o", label=f"X, MAE: {mean_residual_x*1e3:.2f} μm")
            # plt.scatter(diff_y_mean["y"], diff_y_mean["diff_y_mean"], lw=0.8, marker="^", label=f"Y, MAE: {mean_residual_y*1e3:.2f} μm", c='r')
            plt.plot(diff_y_mean["y"], diff_y_mean["diff_y_mean"], lw=1, c='r', linestyle='--', marker="^", label=f"Y, MAE: {mean_residual_y*1e3:.2f} μm")

            # plt.scatter(data['x'], data['x'] - data['cal_X'], lw=0.8, marker="o", label="X", c='b')
            # plt.scatter(data['y'], data['y'] - data['cal_Y'], lw=0.8, marker="^", label="Y", c='r')

            # plt.scatter(data[mask_x]['x'], data[mask_x]['cal_X'], lw=0.8, marker="o", label="X", c='b')
            # plt.scatter(data[mask_y]['y'], data[mask_y]['cal_Y'], lw=0.8, marker="^", label="Y", c='r')
            # plt.ylim([-(cal_range+1), cal_range+1])
            plt.ylim([-1, 1])
            plt.ylabel("Residuals [mm]", fontsize=20)
            plt.xlabel("Wire position [mm]", fontsize=20)
            # plt.plot(data["x"], data["cal_X"], lw=1, c='b')
            # plt.plot(data["y"], data["cal_Y"], lw=1, c="r")

        # plt.plot(x_dummy, y_dummy, c="k", lw=1, ls="-")
        plt.legend(fontsize=14, loc="upper left")
        # plt.title("Linear calibration result")
        plt.xlabel("Wire position [mm]")
        plt.xlim([-(cal_range+1), cal_range+1])
        plt.xticks(range(-8, 9, 4))
        # plt.xticks(range(-cal_range, cal_range+1, 3))
        # plt.yticks(range(-10, 11, 5))
        # plt.gca().set_aspect("equal")
        plt.tight_layout()
    plt.savefig(
        f"{current_DOS}_{target_freq}MHz_fitting_sensitivity.png",
        format="png",
        dpi=500,
        bbox_inches="tight",
    )
    plt.close()
    # plt.show()
    # os._exit(1)
    x_dummy = [np.arange(-max_point, max_point + step, step)] * number_interval
    y_dummy = [
        i
        for i in np.arange(-max_point, max_point + step, step)
        for _ in range(number_interval)
    ]

    tb_dataprocessing.PlotSettings()
    """
    2D mapping
    """
    plt.figure(3, figsize=(12, 4))
    for i, fit in enumerate(cal_method):
        plt.subplot(1, len(cal_method), i + 1)
        plt.suptitle(r"$S_{x=y}$" + f" at {target_freq} MHz", fontsize=22, fontweight='bold')
        if fit == 1:
            plt.title("Linear estimation")
        elif fit == 3:
            plt.title("3rd-order polynomial")
        elif fit == 5:
            plt.title("5th-order polynomial")
        elif fit == "2D-3rd":
            plt.title("2D polynomial")
        cal_x, cal_y = optimizer.optimized_func(data, Wanted_data, cal_range, fit)
        # cal_x_dia, cal_y_dia = optimized_func(data['xDia'], data['yDia'])
        data["cal_X"], data["cal_Y"] = cal_x, cal_y
        cal_offset = data[(data["x"] == 0) & (data["y"] == 0)][["cal_X", "cal_Y"]]

        # data["cal_X"], data["cal_Y"] = (
        #     data["cal_X"] - cal_offset["cal_X"].values,
        #     data["cal_Y"] - cal_offset["cal_Y"].values,
        # )

        plt.scatter(x_dummy, y_dummy, s=40, marker=".", edgecolor="b")
        plt.scatter(
            data["cal_X"],
            data["cal_Y"],
            s=30,
            marker="o",
            facecolor="none",
            edgecolors="r",
        )
        # plt.title("Linear calibration result")
        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.xlim([-max_point - step, max_point + step])
        plt.ylim([-max_point - step, max_point + step])
        plt.gca().set_aspect("equal")
        plt.tight_layout()
        # plt.ylabel("K$_{x, y}$ X DOS ($\Delta/\Sigma$)")
        plt.grid()
    plt.close()
    # plt.show()
    '''
    24. 01. 22 기준 savefile name 수정 필요
    '''
    #     plt.savefig(f'{target_freq}MHz_{sensitivity.strip('$')}2D polynomial.png',
    #     format='png',
    #     dpi=1000,
    # bbox_inches='tight')

    # %%
    # fig1 = plt.figure(figsize=(12, 4))
    '''
    2D error distribution
    '''
    # fig1 = plt.figure(figsize=(8,7))
    if fit_ver == 2:
        # plt.rcParams["axes.titlesize"] = 60
        # plt.rcParams["font.size"] = 20
        fig1 = plt.figure(figsize=(22, 6))
        fig1.suptitle(f"Error distribution of {current_DOS} approach" + f" at {target_freq} MHz", fontsize=32, y=0.90, fontweight='bold')
    else:
        fig1 = plt.figure(figsize=(16, 6))
        fig1.suptitle(f"Error distribution of {current_DOS} approach" + f" at {target_freq} MHz", fontsize=32, y=0.90, fontweight='bold')
    # fig1.set_tight_layout(True)
    for i, fit in enumerate(cal_method):
        # fig = plt.figure(10+i)
        cal_x, cal_y = optimizer.optimized_func(data, Wanted_data, cal_range, fit)
        data["cal_X"], data["cal_Y"] = cal_x, cal_y
        cal_offset = data[(data["x"] == 0) & (data["y"] == 0)][["cal_X", "cal_Y"]]
        # print(cal_offset['cal_X'].values)
        # os._exit()
        '''
        offset
        '''
        # data["cal_X"], data["cal_Y"] = (
        #     data["cal_X"] - cal_offset["cal_X"].values,
        #     data["cal_Y"] - cal_offset["cal_Y"].values,
        # )

        # '''
        # 2D dot mapping
        # '''
        # plt.figure(3, figsize=(12, 4))
        # plt.subplot(1, len(cal_method), i + 1)
        # # plt.suptitle(r"$S_{x=y}$" + f" @ {target_freq} MHz", fontsize=22, fontweight='bold')
        # if fit == 1:
        #     plt.title("Linear estimation")
        # elif fit == 3:
        #     plt.title("3rd-order polynomial")
        # elif fit == 5:
        #     plt.title("5th-order polynomial")
        # elif fit == "2D-3rd":
        #     plt.title("2D polynomial")

        # plt.scatter(x_dummy, y_dummy, s=40, marker=".", edgecolor="b")
        # plt.scatter(
        #     data["cal_X"],
        #     data["cal_Y"],
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

        """
        2D color plotting
        """
        vmin = 0
        vmax = 1
        x_values = np.arange(
            max_point, -max_point - step, -step
        )  # data['x'].to_numpy()
        y_values = x_values  # data['y'].to_numpy()
        cal_XX, cal_YY = data["cal_X"].values.reshape(
            len(x_values), len(x_values)
        ), data["cal_Y"].values.reshape(len(x_values), len(x_values))

        x, y = np.meshgrid(x_values, y_values)
        error_xx, error_yy = x - cal_XX, y - cal_YY
        # z = abs(error_xx) + abs(error_yy)
        z = np.sqrt(error_xx**2 + error_yy**2)  # * 10**3
        if "2D-3rd" in cal_method:
            ax2 = fig1.add_subplot(1, 4, i + 1, aspect="equal")
        else:
            ax2 = fig1.add_subplot(1, 3, i + 1, aspect="equal")
        # fig1.subplots_adjust(left=0.7, right=0.9)
        # plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
        if fit == 3:
            ax2.set_title("3rd polynomial fitting", x=0.5, y= 1.1, fontsize=22)
        elif fit == 1:
            ax2.set_title("Linear fitting", x=0.5, y= 1.1, fontsize=22)
        elif fit == 5:
            ax2.set_title("5th polynomial fitting", x=0.5, y= 1.1, fontsize=22)
        elif fit == "2D-3rd":
            ax2.set_title("2D 3rd-polynomial fitting", x=0.5, y= 1.1, fontsize=22)

        cs = ax2.contourf(
            x, y, z, 30, cmap="jet", vmin=vmin, vmax=vmax
        )  # , vmin=vmin, vmax=vmax
        cs2 = ax2.contour(cs, levels=[0.1], colors="yellow")
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.1)
    
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cs.cmap, norm=cs.norm)
        # cbar.set_ticks([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, vmax])
        # cs2 = ax2.contour(cs, levels=cs.levels[::2], colors='black')
        cbar.add_lines(cs2)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        cbar.set_ticks([0, 0.1, 0.25, 0.5, 0.75, 1.00])
        cbar.set_label("Error [mm]", rotation=270, labelpad=32)
        ax2.clabel(cs2, fmt="%2.1f", colors="y", fontsize=16)
        ax2.set_xlabel("X [mm]", fontsize = 22)
        ax2.set_ylabel("Y [mm]", fontsize = 22)
        ax2.set_xlim([-cal_range, cal_range])
        ax2.set_ylim([-cal_range, cal_range])
        ax2.set_xticks(range(-cal_range, cal_range+1, 4))
        ax2.set_yticks(range(-cal_range, cal_range+1, 4))
        ax2.grid()
        plt.tight_layout()
        # plt.grid()

        '''
        '''
        coeffs = optimizer.get_fit_coeffs(fit)
        if coeffs:
            coeffs_data.append([
                fit, 
                *coeffs['poptx'], 
                # cal_offset['cal_X'].values[0],
                *coeffs['popty'], 
                # cal_offset['cal_Y'].values[0],
            ])

        poptx_length = len(coeffs['poptx'])
        popty_length = len(coeffs['popty'])
        coeffs_df = pd.DataFrame(coeffs_data, columns=[
            'fit_num',
            *[f'K_{poptx_length-i-1}' for i in range(poptx_length)],
            # 'offset_X',
            *[f'L_{popty_length-i-1}' for i in range(popty_length)],
            # 'offset_Y',
        ])

        coeffs_df.to_csv(f'{current_DOS}_fitting_coefficients.csv', index=False)
        print(f'Saved {current_DOS}_fitting_coefficients.csv')

        data.to_csv(f'{current_DOS}_Calibrated_data.csv', index=False)

    plt.savefig(
        f"{current_DOS}_{target_freq}MHz_" + sensi_str + "_2D_colormap.png",
        format="png",
        dpi=500,
        bbox_inches="tight",
    )

    plt.close()
    # plt.show()
    # %%
    error_dict = {}
    range_values = np.arange(step, cal_range + step, step)
    # errors_all = {1: [], 3:[], 5:[], '2D-3rd': []}
    errors_all = dict.fromkeys(cal_method, [])
    print(errors_all)
    # os.exit()
    errors_std, errors_se, errors_mean, errors_rms = {}, {}, {}, {}

    
    """
    범위에따른 에러 그래프
    """
    error_dict, errors_all = optimizer.ErrorWrtRange(data, Wanted_data, cal_range, step, error_dict, errors_all, cal_method)

    print(error_dict)
    # print(np.std(error_dict))
    # sample_size = len(next(iter(errors_all.values())))
    # for fit, errors in errors_all.items():
    #     # print(len(errors))
    #     errors_std[fit] = np.std(errors, axis=0)
    #     errors_se[fit] = errors_std[fit] / np.sqrt(sample_size)
    #     errors_mean[fit] = np.mean(errors, axis=0)
    #     errors_rms[fit] = np.sqrt(np.mean(np.array(errors) ** 2, axis=0))

    plt.figure(99 + fit_ver)
    markers = ["^", "s", "D", ".", "<"]
    p_color = ["r", "b", "magenta", "g", "grey"]
    for i, (fit, error_list) in enumerate(error_dict.items()):
        plt.plot(
            range_values,
            error_list,
            label=f"n = {fit}",
            marker=markers[2 - i],
            c=p_color[i],
        )

    # Find the maximum x where error is less than or equal to 0.10
    # max_x = np.max(np.array(range_values)[np.array(error_list) <= 0.10])
    # if i < 2:
    #     y_value_at_3 = error_list[np.where(np.array(range_values) == 3.0)[0][0]] # Extracting the y-value at x=3
    #     plt.annotate(f"{y_value_at_3:.2f}", (3, y_value_at_3), textcoords="offset points", xytext=(-2,-40), color=p_color[i], ha='right', arrowprops=dict(arrowstyle="->", color=p_color[i]))

    # # Annotation for x=4
    # y_value_at_3_5 = error_list[np.where(np.array(range_values) == 3.5)[0][0]] # Extracting the y-value at x=4
    # plt.annotate(f"{y_value_at_3_5:.2f}", (3.5, y_value_at_3_5), textcoords="offset points", xytext=(14,10), color=p_color[i], ha='right')

    # Annotation for x=4
    # y_value_at_5 = error_list[np.where(np.array(range_values) == 5.0)[0][0]] # Extracting the y-value at x=4
    # plt.annotate(f"{y_value_at_5:.2f}", (5, y_value_at_5), textcoords="offset points", xytext=(80,-23), color=p_color[i], ha='right', arrowprops=dict(arrowstyle="->", color=p_color[i]))

    # plt.axvline(3.0, color='gray', linestyle='--')
    # plt.axvline(4.0, color='gray', linestyle='--')
    plt.suptitle(f"Average error response for {current_DOS}", fontsize=26, fontweight='bold')
    # plt.title(f"Averaged error response for {current_DOS}" + f" @ {target_freq} MHz", fontsize=22, fontweight='bold', pad=20)
    plt.axhline(100, color="gray", linestyle="--")
    plt.xlabel("wire movement plane [mm²]", fontsize=24)
    plt.ylabel("Average error [\u03bcm]", fontsize=24)
    # plt.ylabel(u"\u03bcs")
    labels = [fr"±{i}$\times${i}" for i in range(2, cal_range+1, 2)]
    plt.xticks(range(2, cal_range+1, 2), labels)
    # y_ticks = np.arange([range(0, 601, 100)])
    plt.yticks(range(0, 501, 100))
    # plt.ylim(0, 0.3)
    legend = plt.legend(fontsize=14, title="Fitting order n", loc='upper left')
    plt.setp(legend.get_title(), fontsize=16)
    plt.grid()
    plt.savefig(
        f"{current_DOS}_{target_freq}MHz_" + sensi_str + "_Error_response.png",
        format="png",
        dpi=350,
        bbox_inches="tight",
    )
    print("save completed...")

    # plt.show()
    plt.close()

"""
3D plotting
        fig = plt.figure(10+i)
        ax = fig.add_subplot(111)
        cal_x, cal_y = tb_dataprocessing.optimized_func(data, Wanted_data, cal_range, fit)
        data['cal_X'], data['cal_Y']  = cal_x, cal_y

        # Scatter plots
        ax.scatter(data['x'], data['y'], marker='o', fc='none', edgecolors='r', lw=1, s=50)
        ax.scatter(cal_x, cal_y, marker='4', c='blue', s=50)

        ax.set_xlabel('X [mm]', fontsize=14)
        ax.set_ylabel('Y [mm]', fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')

        # Adjusting the legend
        legend = ax.legend(['Wire', 'Measured'], loc='upper left', bbox_to_anchor=(0.64,1.20))
        legend.get_frame().set_edgecolor('black')


        x_values = np.arange(max_point, -max_point-step, -step) #data['x'].to_numpy()
        y_values = x_values #data['y'].to_numpy()
        # len(x_values)
        cal_XX, cal_YY = cal_x.reshape(len(x_values), len(x_values)), cal_y.reshape(len(x_values), len(x_values))
        # Convert Series to Numpy arrays

        x, y = np.meshgrid(x_values, y_values)    
        error_xx, error_yy = x - cal_XX, y - cal_YY
        # z = abs(error_xx) + abs(error_yy)
        z = np.sqrt( error_xx **2 + error_yy ** 2)

        # Create the plot
        # fig = plt.figure(10+i)
        ax = fig.add_subplot(111, projection='3d')
        if fit == 3:
            fig.suptitle("3rd polynomial fitting", fontsize=16, fontweight='bold', x=0.57)
        elif fit == 1:
            fig.suptitle("Linear fitting", fontsize=16, fontweight='bold', x=0.57)
        elif fit == 5:
            fig.suptitle("5th polynomial fitting", fontsize=18, fontweight='bold', x=0.5)
        # ax.contour(x, y, z, level=20, colors="k", linewidths=1) , vmin=vmin, vmax=vmax
        surf = ax.plot_surface(x, y, z, cmap='jet', rstride=1, cstride=1, antialiased=True, vmin=vmin, vmax=vmax)# , vmin=vmin, vmax=vmax
        ax.set_xlabel('X [mm]', labelpad=3)

        # ax.xaxis.majorTicks[0].set_pad(15)
        ax.set_ylabel('Y [mm]', labelpad=3)
        
        ax.set_zticks([0.2, 0.4, 0.6])

        ax.view_init(elev=50)
        ax.yaxis.set_ticks_position('top')
        plt.close()
"""
# %%
