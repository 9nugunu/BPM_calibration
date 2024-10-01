import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import TestBench_data_processing as tb_processing

# 가정된 데이터 프레임 생성
data = pd.DataFrame({
    'X': [-57, -52, -47, -42, -37, -32],
    'Y': [2.410777018, 1.875337933, 2.81632831, 1.671984877, 3.357473522, 6.278135093]
})

data = pd.DataFrame({
    # 'X': ["1port", "2port", "3port", "4port"],
    'X': ["Top", "Bottom", "Right", "Left"],
    'Y': [-41.289, -44.094, -41.888, -42.583]
})
# 3port	-44.094
# 2port	-42.583
# 1port	-41.289
# 4port	-41.888

# 240423
# 1 -41.527
# 2 -42.431
# 3 -44.995
# 4 -43.917

# 데이터 플로팅
plt.figure(figsize=(8, 6))
tb_processing.PlotSettings()
plt.tight_layout()
plt.grid()
plt.plot(data['X'], data['Y'], marker='o', c='b')
for i, (x, y) in enumerate(zip(data['X'], data['Y'])):
    plt.text(x, y, f"{y:.2f}", fontsize=12, ha='left', va='bottom')
    continue
    if i < 2:
        plt.text(x, y, f"{y:.2f}", fontsize=12, ha='left', va='bottom')
    elif i == 3:
        plt.text(x, y, f"{y:.2f}", fontsize=12, ha='left', va='top')
    elif i == 2:
        plt.text(x, y, f"{y:.2f}", fontsize=12, ha='right', va='top')

# MODIFICATION: Calculating differences
top_bottom_diff = data.loc[data['X'] == 'Top', 'Y'].values[0] - data.loc[data['X'] == 'Bottom', 'Y'].values[0]
right_left_diff = data.loc[data['X'] == 'Right', 'Y'].values[0] - data.loc[data['X'] == 'Left', 'Y'].values[0]

# MODIFICATION: Adding bidirectional arrows
def add_bidirectional_arrow(ax, start, end, text, text_pos):
    # Drawing the arrow
    arrow = FancyArrowPatch(start, end, lw=3, arrowstyle='<->', color='red', mutation_scale=30)
    ax.add_patch(arrow)
    # head_length=0.4, head_width=0.2, widthA=1.0, widthB=1.0, lengthA=0.2, lengthB=0.2, angleA=0, angleB=0, scaleA=None, scaleB=None
    # Adding the text
    ax.text(text_pos[0], text_pos[1], text, fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', alpha=1))

ax = plt.gca()

add_bidirectional_arrow(
    ax,
    (0.5, data.loc[data['X'] == 'Bottom', 'Y'].values[0]),
    (0.5, data.loc[data['X'] == 'Top', 'Y'].values[0]),
    fr"$\Delta_{{Vertical}} $ = {top_bottom_diff:.2f} dB",
    (0.5, (data.loc[data['X'] == 'Top', 'Y'].values[0] + data.loc[data['X'] == 'Bottom', 'Y'].values[0]) / 2)
)

add_bidirectional_arrow(
    ax,
    (2.5, data.loc[data['X'] == 'Left', 'Y'].values[0]),
    (2.5, data.loc[data['X'] == 'Right', 'Y'].values[0]),
    fr"$\Delta_{{Horizontal}} $ = {right_left_diff:.2f} dB",
    (2.5, (data.loc[data['X'] == 'Right', 'Y'].values[0] + data.loc[data['X'] == 'Left', 'Y'].values[0]) / 2)
)
plt.xticks(data['X'])
# plt.yticks([-45, -44, -43, -42, -41])
# plt.title('Four Channel Strength Fluctuations', fontweight='bold')
# plt.xlabel('Input S/G strength [dBm]')
# plt.ylabel('Standard deviation [A.U.]')
# plt.grid(True)
# plt.savefig(
#         f"STD of strength.png",
#         format="png",
#         dpi=500,
#         bbox_inches="tight"
# )


plt.title('Four electrodes transmission characteristic', fontweight='bold')
plt.xlabel('Pick-up electrode')
plt.ylabel('S21 parameters [dB]')
plt.savefig(
        f"Port_S21_parameters.png",
        format="png",
        dpi=350,
        bbox_inches="tight"
)

plt.show()

