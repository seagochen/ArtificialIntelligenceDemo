import numpy as np
from scipy.signal import cwt
from scipy.signal import ricker
import matplotlib.pyplot as plt

# 定义信号频率列表
freq_list = [1, 5, 7, 11, 13, 51, 77, 111]

# 定义信号持续时间
duration = 1  # 1秒

# 定义采样率
sample_rate = 1000  # 1000 Hz

# 计算总采样点数
num_samples = duration * sample_rate

# 生成时间点
t = np.linspace(0, duration, num_samples, endpoint=False)

# 生成信号
signal = np.zeros(num_samples)
for freq in freq_list:
    signal += np.sin(2 * np.pi * freq * t)

# 使用Marr wavelet小波对信号进行处理
widths = np.arange(1, 31)
cwtmatr = cwt(signal, ricker, widths)

low_idx = int(57 / sample_rate * num_samples)
high_idx = int(91 / sample_rate * num_samples)

# 过滤频率
filtered_cwtmatr = cwtmatr.copy()
filtered_cwtmatr[:, low_idx:high_idx+1] = 0

# 绘制原始信号
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Signal')

# 绘制过滤后的信号
plt.subplot(2, 1, 2)
plt.imshow(filtered_cwtmatr, extent=[t[0], t[-1], widths[0], widths[-1]], cmap='PRGn', aspect='auto', vmax=abs(filtered_cwtmatr).max(), vmin=-abs(filtered_cwtmatr).max())
plt.xlabel('Time (s)')
plt.ylabel('Scale')
plt.title('Filtered Signal')

plt.show()