import os, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import TestBench_data_processing as tb_dataprocessing
import matplotlib.ticker as ticker
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable

tb_dataprocessing.PlotSettings()

number_interval = 21

step = 1
max_point = 10
cal_range = 10


Port = "2port/"
Wanted_data = {"X": " X(A)", "Y": " Y(A)"}
sensitivity = r"$S_{\bar{x}}$"
sensi_str = sensitivity.strip("$")

if sensitivity == r"$S_{\bar{x}}$":
    sensi_str = "S_bar_x"

if sensitivity == r"$S_{coupling}$":
    cal_method = [1, 3, 5, "2D-3rd"]  # [1, 3, 5, '2D-3rd']
else:
    cal_method = [1, 3, 5]  # [1, 3, 5, '2D-3rd']


target_freq = "352"


# filename = 'cal_paper__' + '1' + '_4port_01_0.25.csv'
filename = "BPM01_65MHz_20dBm_2port_01_-10to10_100_20231220_182940.csv"
file_dir = (
    "../-5_5_dataset/" + Port + f"BPM01_{target_freq}MHz/"
)  # + filename # 'PAPER_ONLY_0825/' +
file_list = os.listdir(file_dir)
print(type(file_list))
# os.chdir('../' + file_dir)
print(os.getcwd())
print(file_list)
# time.sleep(3)

for file in file_list[2:3]:
    print(file)
    file_path = os.path.join(file_dir, file)
    data = pd.read_csv(file_path, index_col=False)

    # data.drop([' Time', ' Type', ' 1Ch', ' 2Ch',  ' 3Ch', ' 4Ch', ' X(A)', ' X(B)', ' Y(A)', ' Y(B)'], axis=1, inplace=True)
    data.drop([" Time", " Type", " 1Ch", " 2Ch", " 3Ch", " 4Ch"], axis=1, inplace=True)
    data["x"], data["y"] = tb_dataprocessing.add_col_axis(
        number_interval, step, max_point
    )

    # print(data.head())

    plt.figure(1)
    plt.scatter(data[Wanted_data["X"]], data[Wanted_data["Y"]])
    # plt.yticks([0.4, 0.3, 0.2, 0.1, 0.0, -0.1])
    plt.title("measured raw data")
    plt.xlabel("X raw data")
    plt.ylabel("Y raw data")
    # plt.savefig(file_dir+'RAW.png',
    #     format='png',
    #     dpi=1000,
    # bbox_inches='tight')
    # plt.show()

    cal_offset = data[(data["x"] == 0) & (data["y"] == 0)][
        [Wanted_data["X"], Wanted_data["Y"]]
    ]
    x_offset = round(cal_offset[Wanted_data["X"]].values[0] * 1e3, 3)
    y_offset = round(cal_offset[Wanted_data["Y"]].values[0] * 1e3, 3)
    print(rf"x_offset: {x_offset} μm")
    print(f"y_offset: {y_offset} μm")

    mean_same_x = data.groupby("x").mean()
    mean_same_y = data.groupby("y").mean()

    median_x = data.groupby("x").agg(np.median)
    median_y = data.groupby("y").agg(np.median)
    # print(mean_same_x)
    fig3, ax3 = plt.subplots()

    # plt.subplot(111)
    ax3.set_title("Four different sensivitiy curve")
    ax3.plot(
        data[data["x"] == data["y"]]["x"],
        data[data["x"] == data["y"]][Wanted_data["X"]],
        label=r"$S_{x=y}$",
        c="b",
    )
    ax3.plot(
        mean_same_x.index,
        mean_same_x[Wanted_data["X"]],
        label=r"$S_{\bar{x}}$",
        c="r",
        linestyle="--",
    )
    # ax3.plot(
    #     median_x.index,
    #     median_x[Wanted_data["X"]],
    #     label=r"$S_{Med(x)}$",
    #     c="b",
    #     linestyle="dashdot",
    # )
    # plt.legend()
    ax3.grid()
    # plt.subplot(122)
    ax3.scatter(
        data["x"], data[Wanted_data["X"]], label=r"$S_{standard}$", s=8, c="black"
    )
    # plt.scatter(mean_same_y.index, mean_same_y[Wanted_data['Y']], label='mean_same_y', s=10)
    axins = zoomed_inset_axes(
        ax3, 3, loc="lower right", axes_kwargs={"fc": "lightgray"}
    )

    axins.plot(
        data[data["x"] == data["y"]]["x"],
        data[data["x"] == data["y"]][Wanted_data["X"]],
        label=r"$S_{x=y}$",
        c="b",
    )
    axins.plot(
        mean_same_x.index,
        mean_same_x[Wanted_data["X"]],
        label=r"$S_{\bar{x}}$",
        c="r",
        linestyle="--",
    )
    # axins.plot(
    #     median_x.index,
    #     median_x[Wanted_data["X"]],
    #     label=r"$S_{Med(x)}$",
    #     c="b",
    #     linestyle="dashdot",
    # )
    axins.scatter(
        data["x"], data[Wanted_data["X"]], label=r"$S_{standard}$", s=2, c="black"
    )
    axins.set_xlim(7.25, 10.5)
    axins.set_ylim(0.55, 0.78)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.grid()

    mark_inset(ax3, axins, loc1=2, loc2=1, ec="0.5")
    ax3.set_xlabel("X position [mm]")
    ax3.set_ylabel(r"$\Delta/\Sigma$")
    ax3.legend()
    plt.savefig("Four_diff_sensiti.png", format="png", dpi=600, bbox_inches="tight")

    # plt.subplot(223)
    # plt.scatter(data['y'], data[Wanted_data['Y']], label='on_axis', s=10)
    # # plt.scatter(mean_same_y.index, mean_same_y[Wanted_data['Y']], label='mean_same_y')
    # plt.legend()

    # plt.subplot(224)
    # plt.scatter(data['y'], data[Wanted_data['Y']], label='on_axis', s=10)
    # # plt.scatter(mean_same_y.index, mean_same_y[Wanted_data['Y']], label='mean_same_y')
    # plt.legend()

    # print(data.head())
    x_dummy = np.arange(-max_point, max_point + step, step)
    y_dummy = x_dummy
    # print(mean_same_x)

    """
    선형피팅 Sensitivity 출력
    """
    plt.figure(figsize=(10, 6))
    for i, fit in enumerate(cal_method):
        plt.subplot(1, len(cal_method), i + 1)
        plt.grid()
        if fit == 1:
            plt.title("Linear estimation")
        elif fit == 3:
            plt.title("3rd-order polynomial")
        elif fit == 5:
            plt.title("5th-order polynomial")
        elif fit == "2D-3rd":
            plt.title("2D polynomial")
        cal_x, cal_y = tb_dataprocessing.optimized_func(
            data, Wanted_data, cal_range, fit
        )
        # cal_x_dia, cal_y_dia = optimized_func(data['xDia'], data['yDia'])
        data["cal_X"], data["cal_Y"] = cal_x, cal_y

        mean_same_x = data.groupby("x").mean()["cal_X"]
        mean_same_y = data.groupby("y").mean()["cal_Y"]

        plt.scatter(
            mean_same_x.index,
            mean_same_x.values,
            lw=0.8,
            marker="^",
            facecolor="none",
            edgecolor="b",
        )
        plt.scatter(
            mean_same_y.index,
            mean_same_y.values,
            marker=",",
            facecolor="none",
            edgecolors="r",
        )
        plt.plot(x_dummy, y_dummy, c="k", lw=0.5, ls="-")
        plt.legend(["$X_{0}$", "$Y_{0}$"], loc="upper left")
        # plt.title("Linear calibration result")
        plt.xlabel("Wire position [mm]")
        plt.ylabel("Predicted position [mm]")
        plt.xlim([-max_point, max_point])
        plt.ylim([-max_point, max_point])
        plt.gca().set_aspect("equal")
        plt.tight_layout()
    plt.savefig(
        f"{target_freq}MHz_fitting_sensitivity.png",
        format="png",
        dpi=600,
        bbox_inches="tight",
    )
    # plt.ylabel("K$_{x, y}$ X DOS ($\Delta/\Sigma$)")
    # plt.grid()

    x_dummy = [np.arange(-max_point, max_point + step, step)] * number_interval
    y_dummy = [
        i
        for i in np.arange(-max_point, max_point + step, step)
        for _ in range(number_interval)
    ]

    # print(x_dummy)
    # print(len(y_dummy))
    """
    2D mapping
    """
    plt.figure(figsize=(12, 4))

    
    for i, fit in enumerate(cal_method):
        plt.subplot(1, len(cal_method), i + 1)
        plt.suptitle(r"$S_{x=y}$" + f" @ {target_freq} MHz", fontsize=16)
        if fit == 1:
            plt.title("Linear estimation")
        elif fit == 3:
            plt.title("3rd-order polynomial")
        elif fit == 5:
            plt.title("5th-order polynomial")
        elif fit == "2D-3rd":
            plt.title("2D polynomial")
        cal_x, cal_y = tb_dataprocessing.optimized_func(
            data, Wanted_data, cal_range, fit
        )
        # cal_x_dia, cal_y_dia = optimized_func(data['xDia'], data['yDia'])
        data["cal_X"], data["cal_Y"] = cal_x, cal_y
        cal_offset = data[(data["x"] == 0) & (data["y"] == 0)][["cal_X", "cal_Y"]]

        data["cal_X"], data["cal_Y"] = (
            data["cal_X"] - cal_offset["cal_X"].values,
            data["cal_Y"] - cal_offset["cal_Y"].values,
        )

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
    #     plt.savefig(f'{target_freq}MHz_{sensitivity.strip('$')}2D polynomial.png',
    #     format='png',
    #     dpi=600,
    # bbox_inches='tight')

    # %%
    # fig1 = plt.figure(figsize=(12, 4))
    fig1 = plt.figure(figsize=(16, 4))
    # fig1 = plt.figure(figsize=(8,7))
    fig1.suptitle(f"{sensitivity} case" + f" @ {target_freq} MHz", fontsize=16, y=0.92)
    # fig1.set_tight_layout(True)
    for i, fit in enumerate(cal_method):
        """
        3D dimension plotting position
        """

        """
        2D color plotting
        """
        # fig = plt.figure(10+i)
        cal_x, cal_y = tb_dataprocessing.optimized_func(
            data, Wanted_data, cal_range, fit
        )
        data["cal_X"], data["cal_Y"] = cal_x, cal_y
        cal_offset = data[(data["x"] == 0) & (data["y"] == 0)][["cal_X", "cal_Y"]]

        data["cal_X"], data["cal_Y"] = (
            data["cal_X"] - cal_offset["cal_X"].values,
            data["cal_Y"] - cal_offset["cal_Y"].values,
        )

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
            ax2.set_title("3rd polynomial fitting", fontsize=14, x=0.5)
        elif fit == 1:
            ax2.set_title("Linear fitting", fontsize=14, x=0.5)
        elif fit == 5:
            ax2.set_title("5th polynomial fitting", fontsize=14, x=0.5)
        elif fit == "2D-3rd":
            ax2.set_title("2D 3rd-polynomial fitting", fontsize=14, x=0.5)

        cs = ax2.contourf(
            x, y, z, 100, cmap="jet", vmin=vmin, vmax=vmax
        )  # , vmin=vmin, vmax=vmax
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cs.cmap, norm=cs.norm)
        # cbar.set_ticks([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, vmax])
        # cs2 = ax2.contour(cs, levels=cs.levels[::2], colors='black')
        cs2 = ax2.contour(cs, levels=[0.1], colors="yellow")
        cbar.add_lines(cs2)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        cbar.set_label("Error [mm]", rotation=270, labelpad=15)
        ax2.clabel(cs2, fmt="%2.1f", colors="y", fontsize=14)
        ax2.set_xlabel("X [mm]")
        ax2.set_ylabel("Y [mm]")
        ax2.set_xlim([-cal_range, cal_range])
        ax2.set_ylim([-cal_range, cal_range])
        plt.tight_layout()
    plt.savefig(
        f"{target_freq}MHz_" + sensi_str + "_2D_colormap.png",
        format="png",
        dpi=600,
        bbox_inches="tight",
    )
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
for file in file_list:
    # errors_all = dict.fromkeys(cal_method, [])
    file_path = os.path.join(file_dir, file)
    data = pd.read_csv(file_path, index_col=False)

    data.drop([" Time", " Type", " 1Ch", " 2Ch", " 3Ch", " 4Ch"], axis=1, inplace=True)
    data["x"], data["y"] = tb_dataprocessing.add_col_axis(
        number_interval, step, max_point
    )
    error_dict, errors_all = tb_dataprocessing.ErrorWrtRange(
        data, Wanted_data, cal_range, step, error_dict, errors_all, cal_method
    )

    print(error_dict)
    sample_size = len(next(iter(errors_all.values())))
    for fit, errors in errors_all.items():
        # print(len(errors))
        errors_std[fit] = np.std(errors, axis=0)
        errors_se[fit] = errors_std[fit] / np.sqrt(sample_size)
        errors_mean[fit] = np.mean(errors, axis=0)
        errors_rms[fit] = np.sqrt(np.mean(np.array(errors) ** 2, axis=0))

plt.figure()
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
plt.title(f"{sensitivity} case" + f" @ {target_freq} MHz")
plt.axhline(100, color="gray", linestyle="--")
plt.xlabel("wire movement range [mm]")
plt.ylabel("Average error [\u03bcm]")
# plt.ylabel(u"\u03bcs")
plt.xticks(range_values)
# y_ticks = np.arange(0, 201, 25)
# plt.yticks(y_ticks)
# plt.ylim(0, 0.3)
plt.legend()
plt.savefig(
    f"{target_freq}MHz_" + sensi_str + "_Error_response.png",
    format="png",
    dpi=600,
    bbox_inches="tight",
)
plt.show()


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
