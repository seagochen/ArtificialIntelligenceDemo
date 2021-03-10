import pandas

data = {
    'state': ['a', 'b', 'c', 'd'],
    'data1': [1, 2, 3, 4],
    'data2': [3, 4, 5, 6]
}

# append new column
frame = pandas.DataFrame(data)
frame['data3'] = frame['data1'] + frame['data2']
print(frame)

# delete column
# del frame['data3']
# frame = frame.drop(columns=['data3'])
frame['data3'] = 100
print(frame)

# delete row
# frame = frame.drop(index=[2])
# print(frame)

# modify row
# frame.iloc[0] = 10
# print(frame)+
data = {'state': 'e', 'data1': 1000, 'data2': 2000}
row = pandas.Series(data)
frame = frame.append(row, ignore_index=True)
print(frame)

corr = frame.corr()
print(corr)