base = 'reports/'
names = [str(i) + '.txt'  for i in range(1, 4)]

for name in names:
    data = []

    for prefix in ('s', 'p'):
        with open(base + prefix + name) as f:
            data.append(f.read())

    print('Equal sizes: ', len(data[0]) == len(data[1]),
                           len(data[0]), len(data[1]))
    print('Equal content: ', data[0] == data[1])
