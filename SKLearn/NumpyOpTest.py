import numpy

origin_data = [1, 2, 3, 4, 5, 6]
arr1 = numpy.array(origin_data)

# 对 numpy 元素的矢量计算进行测试
# print(arr1 * arr1)
# print(arr1 - arr1)

# out:
# [ 1  4  9 16 25 36]
# [0 0 0 0 0 0]

mat1 = numpy.array([[1, 2, 3]])
mat2 = numpy.array([2, 2, 2])
# print(mat1 * mat2)

# out: 生成的是元素计算结果
# [[2 4 6]]

# 矩阵计算
mat3 = numpy.matmul(mat1, mat2)
# print(mat3)

# out:
# [12]

mat3 = mat1.dot(mat2)
# print(mat3)

# out:
# [12]

# 寻址
array = numpy.empty(32, numpy.int32)
for i in range(0, 32):
    array[i] = i + 1

array = array.reshape([8, 4])
print(array)

print("then...")

out = array[[ 1, 5, 6, 2]][:, [0, 3, 1, 2]]
print(out)

# 转置矩阵
print(array.T)