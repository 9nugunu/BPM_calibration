import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import TestBench_data_processing as tb_processing

# 가정된 데이터 프레임 생성
data = pd.DataFrame({
    'X': [0, 0, 1, 1],  # X 위치를 0과 1로 조정하여 Top과 Bottom, Right과 Left를 같은 Y 라인에 놓음
    'Y': [-41.289, -44.094, -42.583, -41.888],
    'Label': ['Top', 'Bottom', 'Right', 'Left']  # 라벨 데이터 추가
})

# 데이터 플로팅
plt.figure(figsize=(8, 6))
tb_processing.PlotSettings()
plt.tight_layout()
plt.grid()

# X축 레이블 제거
# plt.xticks(data['X'])  # 이 줄을 삭제

# 데이터와 라벨 플로팅
for i, (x, y, label) in enumerate(zip(data['X'], data['Y'], data['Label'])):
    plt.plot(x, y, marker='o', c='b')
    plt.text(x, y, f"{y:.2f}\n{label}", fontsize=12, ha='center', va='bottom')

# 나머지 코드는 동일하게 유지하되, X 좌표를 조정
# ...

plt.show()
