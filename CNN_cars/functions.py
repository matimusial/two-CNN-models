import copy
import numpy as np
def mapdata(data):
    dict = {
        'vhigh': 0.0,
        'high': 1.0,
        'med': 2.0,
        'low': 3.0,
        '2': 0.0,
        '3': 1.0,
        '4': 2.0,
        '5more': 3.0,
        'more': 2.0,
        'small': 0.0,
        'big': 2.0,
    }


    if len(data.shape)>1:
        result = np.array([[dict[slovo] for slovo in wiersz] for wiersz in data])
    else:
        result = np.array([dict[slovo] for slovo in data])

    return result


def normalize(data):
    data_normalized = copy.deepcopy(data)
    for j in range(data.shape[1]):
        min = np.min(data[:,j])
        max = np.max(data[:,j])
        curr_data = data[:,j]
        data_normalized[:,j]=((curr_data-min)/(max-min))

    return data_normalized