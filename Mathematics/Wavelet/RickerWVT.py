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

# 计算小波函数数组
wavelet_1 = base_wavelet_ricker(1, num_samples)
wavelet_5 = base_wavelet_ricker(5, num_samples)
wavelet_10 = base_wavelet_ricker(10, num_samples)

# 使用小波函数进行卷积
result_1 = np.convolve(signal, wavelet_1, mode='same')
result_5 = np.convolve(signal, wavelet_5, mode='same')
result_10 = np.convolve(signal, wavelet_10, mode='same')

# 绘制小波变换结果
plt.plot(time, result_1, label='1Hz')
plt.plot(time, result_5, label='5Hz')
plt.plot(time, result_10, label='10Hz')

plt.legend()
plt.show()

# 将得到的结果与小波函数进行卷积，得到原始信号
result_1 = np.convolve(result_1, wavelet_1, mode='same')
result_5 = np.convolve(result_5, wavelet_5, mode='same')
result_10 = np.convolve(result_10, wavelet_10, mode='same')


# 绘制小波逆变换结果
plt.plot(time, result_1, label='1Hz')
plt.plot(time, result_5, label='5Hz')
plt.plot(time, result_10, label='10Hz')

plt.legend()
plt.show()
