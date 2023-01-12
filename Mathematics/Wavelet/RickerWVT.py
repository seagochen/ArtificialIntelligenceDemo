import numpy as np
from matplotlib import pyplot as plt

def base_wavelet_ricker(frequency, num_samples, sampling_time=1.0):
    t = np.linspace(-sampling_time/2, sampling_time/2, num_samples)
    wavelet = (1.0 - 2.0 * np.pi**2 * frequency**2 * t**2) * np.exp(-np.pi**2 * frequency** 2 * t**2)
    return wavelet


# 创建信号和时间数组
num_samples = 1000
duration = 1.0
time = np.linspace(0, duration, num_samples)
signal = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 5 * time) + np.sin(2 * np.pi * 1 * time)

# 绘制原始信号
# plt.plot(time, signal)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()

# 对原始信号的频率进行分析
# spectrum = np.fft.fft(signal) # 计算信号的频谱
# freq = np.fft.fftfreq(signal.size, d=duration/num_samples) # 计算频率数组

# 绘制原始信号的频谱
# plt.plot(freq, np.abs(spectrum))
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.show()

# 分别使用1-16Hz的小波函数对信号进行小波变换
wavelets = []
for i in range(1, 17):
    wavelet = base_wavelet_ricker(i, num_samples, duration)
    wavelets.append(wavelet)

# 使用阈值法，对小波变换后的信号成分进行筛选
threshold = 0.5
results = []
for i in range(len(wavelets)):
    result = np.convolve(signal, wavelets[i], mode='same') # 使用小波函数进行卷积
    result = np.convolve(result, wavelets[i], mode='same') # 使用逆卷积判断信号成分的有效性
    result[np.abs(result) < threshold] = 0
    results.append(result)

# 创建三维坐标轴对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制小波时频图
X, Y = np.meshgrid(time, range(1, 17))
ax.plot_surface(X, Y, np.array(results), cmap='rainbow')

# 设置坐标轴标签
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
ax.set_zlabel('Amplitude')

# 显示图像
plt.show()