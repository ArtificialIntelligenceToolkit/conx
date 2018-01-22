import numpy as np

def make_figure_ground_a():
    ## Letters are 17 rows x 9 cols
    ## leaves off top 4 and bottom 4
    ## from regular gridfont letters
    data = []
    letter = []
    # First, read file
    ## Original data set was
    ## source             -> target
    ## letter, brim, body -> letter, brim, body
    with open("figure_ground_a.dat") as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            if line:
                L = [round(float(v)) for v in line.split(" ")]
                letter.append(L)
            else:
                data.append(letter)
                letter = []
            line = fp.readline()
        if letter:
            data.append(letter)
            letter = []
    ## we just want the letter, brim, body
    dict = make_dict(data)
    ds = make_dataset(dict, data)
    np.save("figure_ground_a.npy", ds)

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

def get_bits(byte):
    lookup = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "A": 10,
        "B": 11,
        "C": 12,
        "D": 13,
        "E": 14,
        "F": 15,
    }
    return ("0000" + (bin(lookup[byte.upper()])[2:]))[-4:]

def bit2pos(bit):
    """
    Given bit code, return list of (x,y) for 
    each pixel to set.

        0  1  2  3  4  5  6  7  8
     0 xx 00 00 00 xx 11 11 11 xx
     1 14 44    32 15 45    33 16
     2 14    xx    15    xx    16
     3 14 32    44 15 33    45 16
     4 xx 02 02 02 xx 03 03 03 xx
     5 17 46    34 18 47    35 19
     6 17    xx    18    xx    19
     7 17 34    46 18 35    47 19
     8 xx 04 04 04 xx 05 05 05 xx
     9 20 48    36 21 49    37 22
    10 20    xx    21    xx    22
    11 20 36    48 21 37    49 22
    12 xx 06 06 06 xx 07 07 07 xx
    13 23 50    38 24 51    39 25
    14 23    xx    24    xx    25
    15 23 38    50 24 39    51 25
    16 xx 08 08 08 xx 09 09 09 xx
    17 26 52    40 27 53    41 28
    18 26    xx    27    xx    28
    19 26 40    52 27 41    53 28
    20 xx 10 10 10 xx 11 11 11 xx
    21 29 54    42 30 55    43 31
    22 29    xx    30    xx    31
    23 29 42    54 30 43    55 31
    24 xx 12 12 12 xx 13 13 13 xx
    """
    if bit == 0:
        return [(0,0), (1,0), (2,0), (3,0), (4,0)]
    elif bit == 1:
        return [(4,0), (5,0), (6,0), (7,0), (8,0)]
    elif bit == 2:
        return [(0,4), (1,4), (2,4), (3,4), (4,4)]
    elif bit == 3:
        return [(4,4), (5,4), (6,4), (7,4), (8,4)]
    elif bit == 4:
        return [(0,8), (1,8), (2,8), (3,8), (4,8)]
    elif bit == 5:
        return [(4,8), (5,8), (6,8), (7,8), (8,8)]
    elif bit == 6:
        return [(0,12), (1,12), (2,12), (3,12), (4,12)]
    elif bit == 7:
        return [(4,12), (5,12), (6,12), (7,12), (8,12)]
    elif bit == 8:
        return [(0,16), (1,16), (2,16), (3,16), (4,16)]
    elif bit == 9:
        return [(4,16), (5,16), (6,16), (7,16), (8,16)]
    elif bit == 10:
        return [(0,20), (1,20), (2,20), (3,20), (4,20)]
    elif bit == 11:
        return [(4,20), (5,20), (6,20), (7,20), (8,20)]
    elif bit == 12:
        return [(0,24), (1,24), (2,24), (3,24), (4,24)]
    elif bit == 13:
        return [(4,24), (5,24), (6,24), (7,24), (8,24)]
    elif bit == 14:
        return [(0,0), (0,1), (0,2), (0,3), (0,4)]
    elif bit == 15:
        return [(4,0), (4,1), (4,2), (4,3), (4,4)]
    elif bit == 16:
        return [(8,0), (8,1), (8,2), (8,3), (8,4)]
    elif bit == 17:
        return [(0,4), (0,5), (0,6), (0,7), (0,8)]
    elif bit == 18:
        return [(4,4), (4,5), (4,6), (4,7), (4,8)]
    elif bit == 19:
        return [(8,4), (8,5), (8,6), (8,7), (8,8)]
    elif bit == 20:
        return [(0,8), (0,9), (0,10), (0,11), (0,12)]
    elif bit == 21:
        return [(4,8), (4,9), (4,10), (4,11), (4,12)]
    elif bit == 22:
        return [(8,8), (8,9), (8,10), (8,11), (8,12)]
    elif bit == 23:
        return [(0,12), (0,13), (0,14), (0,15), (0,16)]
    elif bit == 24:
        return [(4,12), (4,13), (4,14), (4,15), (4,16)]
    elif bit == 25:
        return [(8,12), (8,13), (8,14), (8,15), (8,16)]
    elif bit == 26:
        return [(0,16), (0,17), (0,18), (0,19), (0,20)]
    elif bit == 27:
        return [(4,16), (4,17), (4,18), (4,19), (4,20)]
    elif bit == 28:
        return [(8,16), (8,17), (8,18), (8,19), (8,20)]
    elif bit == 29:
        return [(0,20), (0,21), (0,22), (0,23), (0,24)]
    elif bit == 30:
        return [(4,20), (4,21), (4,22), (4,23), (4,24)]
    elif bit == 31:
        return [(8,20), (8,21), (8,22), (8,23), (8,24)]
    ## lower-left, to upper right
    elif bit == 32:
        return [(0,4), (1,3), (2,2), (3,1), (4,0)]
    elif bit == 33:
        return [(4,4), (5,3), (6,2), (7,1), (8,0)]
    elif bit == 34:
        return [(0,8), (1,7), (2,6), (3,5), (4,4)]
    elif bit == 35:
        return [(4,8), (5,7), (6,6), (7,5), (8,4)]
    elif bit == 36:
        return [(0,12), (1,11), (2,10), (3,9), (4,8)]
    elif bit == 37:
        return [(4,12), (5,11), (6,10), (7,9), (8,8)]
    elif bit == 38:
        return [(0,16), (1,15), (2,14), (3,13), (4,12)]
    elif bit == 39:
        return [(4,16), (5,15), (6,14), (7,13), (8,12)]
    elif bit == 40:
        return [(0,20), (1,19), (2,18), (3,17), (4,16)]
    elif bit == 41:
        return [(4,20), (5,19), (6,18), (7,17), (8,16)]
    elif bit == 42:
        return [(0,24), (1,23), (2,22), (3,21), (4,20)]
    elif bit == 43:
        return [(4,24), (5,23), (6,22), (7,21), (8,20)]
    ## upper-left to lower-right:
    elif bit == 44:
        return [(0,0), (1,1), (2,2), (3,3), (4,4)]
    elif bit == 45:
        return [(4,0), (5,1), (6,2), (7,3), (8,4)]
    elif bit == 46:
        return [(0,4), (1,5), (2,6), (3,7), (4,8)]
    elif bit == 47:
        return [(4,4), (5,5), (6,6), (7,7), (8,8)]
    elif bit == 48:
        return [(0,8), (1,9), (2,10), (3,11), (4,12)]
    elif bit == 49:
        return [(4,8), (5,9), (6,10), (7,11), (8,12)]
    elif bit == 50:
        return [(0,12), (1,13), (2,14), (3,15), (4,16)]
    elif bit == 51:
        return [(4,12), (5,13), (6,14), (7,15), (8,16)]
    elif bit == 52:
        return [(0,16), (1,17), (2,18), (3,19), (4,20)]
    elif bit == 53:
        return [(4,16), (5,17), (6,18), (7,19), (8,20)]
    elif bit == 54:
        return [(0,20), (1,21), (2,22), (3,23), (4,24)]
    elif bit == 55:
        return [(4,20), (5,21), (6,22), (7,23), (8,24)]
    else:
        raise Exception("no such bit number")

def make_letter(bits):
    array = [[0.0 for row in range(25)] for col in range(9)]
    for index in range(len(bits)):
        if bits[index] == "1":
            positions = bit2pos(index)
            for (x,y) in positions:
                array[x][y] = 1.0
    letter = np.array(array)
    letter = letter.swapaxes(0, 1)
    return letter.tolist()
    #return array
    
def read_gridfonts():
    data = []
    labels = []
    with open("gridfonts.dat") as fp:
        line = fp.readline()
        while line:
            if " : " in line or line == "\n":
                line = fp.readline()
                continue
            line = line.strip()
            letter, code = line.split(" ")
            #print(letter, code)
            bits = "".join([get_bits(byte) for byte in code])
            #print(bits)
            data.append(make_letter(bits))
            labels.append(letter)
            line = fp.readline()
    return data, labels

def display_letter(letter):
    for row in range(25):
        for col in range(9):
            print( " X"[int(letter[row][col])], end="")
        print()
    print()

def make_gridfonts():
    data, labels = read_gridfonts()
    np.save("gridfonts.npy", [data, labels])

make_figure_ground_a()
make_gridfonts()
