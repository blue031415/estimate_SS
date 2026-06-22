from ezc3d import c3d
from utils.preprocess import remove_nan
import numpy as np


def read_c3d(path):
    c = c3d(path)
    point_data = remove_nan(c["data"]["points"])  # (XYZ1, num_mark, num_frame)
    data = np.swapaxes(point_data[0:3, :, :], 0, 1)  # (num_mark, XYZ, num_frame)
    return data
