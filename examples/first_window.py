"""第一课：观察同一位参与者走路和站立时的两个信号窗口。"""

from pathlib import Path

import numpy as np
import torch


data = Path(__file__).resolve().parents[1] / "data" / "UCI HAR Dataset"
train = data / "train"
subjects = np.loadtxt(train / "subject_train.txt", dtype=int)
labels = np.loadtxt(train / "y_train.txt", dtype=int)
activities = dict(line.split() for line in (data / "activity_labels.txt").read_text().splitlines())

person = 1
activity_ids = [1, 5]
rows = [np.flatnonzero((subjects == person) & (labels == code))[0] for code in activity_ids]

# 每个文件是一条信号通道，同一行对应同一个 2.56 秒窗口。
channels = []
for signal in ("total_acc", "body_acc", "body_gyro"):
    for axis in ("x", "y", "z"):
        path = train / "Inertial Signals" / f"{signal}_{axis}_train.txt"
        values = np.loadtxt(path, dtype=np.float32, max_rows=max(rows) + 1)
        channels.append(values[rows])

windows = torch.from_numpy(np.stack(channels, axis=1))
print("窗口数 × 通道数 × 采样点数", tuple(windows.shape))
assert windows.shape == (len(rows), 9, 128)
for window, row in zip(windows, rows):
    x = window[0]  # total_acc_x，单位为 g
    print(f"{activities[str(labels[row])]}  人员 {subjects[row]}  原文件第 {row + 1} 行")
    print("  x 轴总加速度前 6 点", x[:6].tolist())
    print(f"  这个窗口的 x 轴标准差 {x.std(correction=0).item():.4f} g")
