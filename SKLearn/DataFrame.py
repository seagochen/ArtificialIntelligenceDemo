import pandas

frame_data = {
    'a': [1, 2, 3, 4],
    'b': ['a', 'b', 'c', 'd'],
    'c': [1.1, 2.2, 3.3, 4.4]
}

frame = pandas.DataFrame(frame_data, index=['A', 'B', 'C', 'D'])
print(frame)

print(frame.a)
print(frame.loc['A'])