import numpy as np
from scipy import signal


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
    abc = []
    for i in range(len(markers)):
        point = np.asarray((df[side + markers[i] + coords[0]][frame],
                            df[side + markers[i] + coords[1]][frame],
                            df[side + markers[i] + coords[2]][frame]))
        abc.append(point)

    return tuple(abc)


def tilt_correct(df, reference_marker, columns_to_correct):
    """
    If the runway is not perfectly aligned there can be a linear trend in one of the axis.
    This function computes the linear trend from a reference marker and applies it to all the columns passed.
    E.g. use left mtp z axis to compute the trend, then subtract the trend from all columns representing the z axis
    of a marker.

    """

    trend = signal.detrend(df[reference_marker]) - df[reference_marker]
    df_tilt_corrected = df.apply(lambda x: x.add(trend, axis=0) if x.name in columns_to_correct else x)
    return df_tilt_corrected


def shift_correct(df, reference_marker, columns_to_correct):
    """
    If the origin is not set to the beginning of the runway (e.g. set to the center) one of the axis will have negative
    values. This functions shifts all the columns to be corrected by the minimum value of the reference marker.
    The reference marker should be the one farther in the back.

    """

    shift = abs(min(df[reference_marker])) if min(df[reference_marker]) < 0 else 0
    df_shift_corrected = df.apply(lambda x: x.add(shift, axis=0) if x.name in columns_to_correct else x)
    return df_shift_corrected
