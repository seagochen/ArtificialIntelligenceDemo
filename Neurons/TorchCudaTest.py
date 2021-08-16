import torch

#返回当前设备索引
# print(torch.cuda.current_device())

#返回GPU的数量
# print(torch.cuda.device_count())

#返回gpu名字，设备索引默认从0开始
# print(torch.cuda.get_device_name(0))

#cuda是否可用
# print(torch.cuda.is_available())

# pytorch 查看cuda 版本
# 由于pytorch的whl 安装包名字都一样，所以我们很难区分到底是基于cuda 的哪个版本。
# print(torch.version.cuda)

# 判断pytorch是否支持GPU加速
# print(torch.cuda.is_available())

# 【PyTorch】查看自己的电脑是否已经准备好GPU加速（CUDA）
# 那么在CUDA已经准备好的电脑上，会输出：cuda:0
# 而在没有CUDA的电脑上则输出：cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
