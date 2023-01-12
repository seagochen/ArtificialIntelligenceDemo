import numpy as np
from scipy.datasets import electrocardiogram
from scipy.signal import resample
import matplotlib.pyplot as plt

# 生成ECG信号
ecg = electrocardiogram() # ECG信号，原始长度为5分钟，采样率为360Hz
fs = 360 # 采样频率

# 重新调整采样率为1024Hz
desired_fs = 1024 # 目标采样频率
ecg = resample(ecg, int(len(ecg) * desired_fs / fs))

# 截取前5s的信号
samples_num = desired_fs * 5 # 5s的采样点数
ecg = ecg[:samples_num]

# 随机生成噪音信号
noise_freq = np.random.randint(200, 231) # 随机生成噪音频率
noise_amp = 0.02 # 噪音幅值
time = np.linspace(0, 5, samples_num)
noise = noise_amp * np.sin(2 * np.pi * noise_freq * time)
ecg_with_noise = ecg + noise

# 在同一界面上下绘制原始信号和噪音信号
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(ecg)
plt.title('Original ECG')
plt.subplot(2, 1, 2)
plt.plot(ecg_with_noise)
plt.title('ECG with Noise')
plt.tight_layout()
plt.show()


def wavelet_ricker(frequency, num_samples, sampling_time=1.0):
    t = np.linspace(-sampling_time/2, sampling_time/2, num_samples)
    wavelet = (1.0 - 2.0 * np.pi**2 * frequency**2 * t**2) * np.exp(-np.pi**2 * frequency** 2 * t**2)
    
    print('Wavelet frequency: {} Hz'.format(frequency))
    print('Wavelet sampling time: {} s'.format(sampling_time))
    print('Wavelet number of samples: {}'.format(num_samples))
    print('Wavelet sampling frequency: {} Hz'.format(1.0 / sampling_time))
    print('Wavelet maximum frequency: {} Hz'.format(1.0 / (2.0 * sampling_time)))
    print('Wavelet minimum frequency: {} Hz'.format(1.0 / (num_samples * sampling_time)))
    
    return wavelet

# 生成小波函数
wavelet = wavelet_ricker(5, 5, 1.0 / desired_fs)

# 使用该小波函数对原始信号进行小波变换
coeffs = np.convolve(ecg, wavelet, mode='same')

# 再次使用该小波函数对小波变换后的信号进行小波逆变换
reconstructed_ecg = np.convolve(coeffs, wavelet, mode='same')

# 绘制原始信号和重构信号
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(ecg)
plt.title('Original ECG')
plt.subplot(2, 1, 2)
plt.plot(reconstructed_ecg)
plt.title('Reconstructed ECG')
plt.tight_layout()
plt.show()

# 开始对有噪音的信号进行去噪
coeffs = np.convolve(ecg_with_noise, wavelet, mode='same')

# 将小波变换后的系数中的噪音系数置为0
coeffs[abs(coeffs) < 0.1] = 0

# 再次使用该小波函数对小波变换后的信号进行小波逆变换
reconstructed_ecg = np.convolve(coeffs, wavelet, mode='same')

# 绘制原始信号和重构信号
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(ecg_with_noise)
plt.title('ECG with Noise')
plt.subplot(2, 1, 2)
plt.plot(reconstructed_ecg)
plt.title('Denoised ECG')
plt.tight_layout()
plt.show()