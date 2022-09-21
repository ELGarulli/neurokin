import numpy as np


def get_angle(a, b, c):
    """
    Get angle between 3 points in a 3d space
    :param a: array with 3d coordinates of a
    :param b: array with 3d coordinates of b
    :param c: array with 3d coordinates of c
    :return: angle in degree
    """
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def get_points(df, markers, side, frame):
    coords = ["_x", "_y", "_z"]
    a = np.asarray((df[side + markers[0] + coords[0]][frame],
                    df[side + markers[0] + coords[1]][frame],
                    df[side + markers[0] + coords[2]][frame]))
    b = np.asarray((df[side + markers[1] + coords[0]][frame],
                    df[side + markers[1] + coords[1]][frame],
                    df[side + markers[1] + coords[2]][frame]))
    c = np.asarray((df[side + markers[2] + coords[0]][frame],
                    df[side + markers[2] + coords[1]][frame],
                    df[side + markers[2] + coords[2]][frame]))
    return a, b, c
