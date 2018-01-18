import numpy as np

def gridfonts():
    data = []             
    letter = []
    with open("gridfonts.dat") as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            if line:
                letter.append([round(float(v)) for v in line.split(" ")])
            else:
                data.append(letter)
                letter = []
            line = fp.readline()
        if letter:
            data.append(letter)
            letter = []
    return data

def display_font(font):
    def ascii(v):
        if v == 0:
            return " "
        elif v == 1:
            return "X"
        else:
            raise Exception("invalid value")
    for row in range(17):
        for col in range(9):
            print(ascii(font[row * 9 + col]), end="")
        print()
    print()

def make_dict(data):
    dict = {}
    for i in range(len(data)):
        vector = tuple(data[i][0])
        if vector not in dict:
            dict[vector] = i
    return dict

def make_dataset(dict, data):
    dataset = []
    for v in dict:
        index = dict[v]
        figure = data[index][1]
        ground = data[index][2]
        dataset.append([v, [figure, ground]])
    return dataset

data = gridfonts()
dict = make_dict(data)
ds = make_dataset(dict, data)
np.save("gridfonts.npy", ds)
