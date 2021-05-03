import numpy
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

# 图像细节调整, 也可以采用类似的方式进行调整
# params={
#    'axes.labelsize': '35',
#    'xtick.labelsize':'27',
#    'ytick.labelsize':'27',
#    'lines.linewidth':2 ,
#    'legend.fontsize': '27',
#    'figure.figsize' : '12, 9'
# }
#
# pyplot.rcParams.update(params)
# 具体配置，可以参考 https://matplotlib.org/3.2.1/tutorials/introductory/customizing.html
#
pyplot.rcParams['axes.labelsize'] = 14  # fontsize of the x any y labels
pyplot.rcParams['xtick.labelsize'] = 12  # fontsize of the tick labels
pyplot.rcParams['ytick.labelsize'] = 12  # fontsize of the tick labels

# seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同，
# 如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
numpy.random.seed(42)

# 创建数据集
# make_moons: Make two interleaving half circles
# sklearn.datasets.make_moons(n_samples=100, *, shuffle=True, noise=None, random_state=None)
#
# Parameters:
# <n_samples> int or two-element tuple, optional (default=100)
# If int, the total number of points generated. If two-element tuple, number of points in each of two moons.
#
# <shuffle> bool, optional (default=True)
# Whether to shuffle the samples.
#
# <noise> double or None (default=None)
# Standard deviation of Gaussian noise added to the data.
#
# <random_state> int, RandomState instance, default=None
# Determines random number generation for dataset shuffling and noise. Pass an int for reproducible
# output across multiple function calls. See Glossary.
#
# Returns:
# X: array of shape [n_samples, 2], numpy.ndarray
# The generated samples.
#
# Y: array of shape [n_samples], numpy.ndarray
# The integer labels (0 or 1) for class membership of each sample.
X, Y = make_moons(n_samples=500, noise=0.30, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 绘制数据
pyplot.plot(X[:, 0][Y == 0], X[:, 1][Y == 0], 'yo', alpha=0.6)
pyplot.plot(X[:, 0][Y == 1], X[:, 1][Y == 1], 'bs', alpha=0.6)
# pyplot.scatter(X[:,0], X[:,1], marker='o', c=Y)
pyplot.show()

# print(X[:, 0])
