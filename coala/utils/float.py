import numpy as np


def rounding(array, num):
    return np.around(np.array(array), num).tolist()
