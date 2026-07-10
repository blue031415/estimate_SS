from ezc3d import c3d
from utils.preprocess import remove_nan


def read_c3d(path, return_labels=False):
    c = c3d(path)
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    point_data = remove_nan(c["data"]["points"])  # (XYZ1, num_mark, num_frame)
    data = point_data[0:3, :, :].transpose(2, 1, 0)  # (num_frame, num_mark, XYZ)
    if return_labels:
        return data, labels[: data.shape[1]]
    return data


if __name__ == "__main__":

    c = c3d("data/07_01.c3d")
    labels = c["parameters"]["POINT"]["LABELS"]["value"]

    for i, label in enumerate(labels):
        print(i, label)
