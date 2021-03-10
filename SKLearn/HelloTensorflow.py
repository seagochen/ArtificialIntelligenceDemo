#引入类库
import tensorflow as tf
import numpy as np
from tensorflow import keras

#使用下述语句来查看tensorflow版本，以下代码都是2.0版的
print(tf.__version__)

#使用array来组织数据整理
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#定义模型model,该模型是具有一个输入(input_shape[1])和一个神经元输出的全连接（Dense）模型。
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
#使用SGD优化器、和均方误差来编译模型。SGD和mean_squared_error后面脑补
model.compile(optimizer='sgd', loss='mean_squared_error')
#开始训练， 500次
model.fit(xs, ys, epochs=500)
#用训练好的model预测10.0，打印其预测值是多少
print(model.predict([10.0]))
