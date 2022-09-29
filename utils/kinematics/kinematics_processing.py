import numpy as np
from scipy import signal


def get_angle_3d(coordinates):
    # TODO check if 3d vs 2d is necessary
    """
    Get angle between 3 points in a 3d space
    :param a: array with 3d coordinates of a
    :param b: array with 3d coordinates of b
    :param c: array with 3d coordinates of c
    :return: angle in degree
    """
    a = coordinates[0]
    b = coordinates[1]
    c = coordinates[2]
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def get_angle_2d(coordinates):
    """
    REF https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
    :param p1:
    :param p2:
    :return:
    """
    a = coordinates[0]
    b = coordinates[1]
    c = coordinates[2]

    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def get_marker_coordinates_names(df_columns_names, markers):
    """
    Returns the names of the columns that contain the name of the marker, to retrieve the 2 or 3 coordinates.
    E.g. lknee, will could return lknee_x, lknee_y, lknee_z.
    :param df_columns_names: dataframe column names with all the markers
    :param markers: markers of interest
    :return:
    """
    abc = []
    for i in range(len(markers)):
        point = [x for x in df_columns_names if markers[i] in x]
        abc.append(point)
    abc.sort()          # courtesy of me dreaming code. makes xyz order assumption more likely, still an assumption.
    return tuple(abc)


def get_marker_coordinate_values(df, a, frame):
    """
    Given a dataframe, a list of column names referring to x, y (and z) of the same marker, and a frame number,
    it returns the corresponding values.
    :param df: dataframe
    :param a: sets of markers names
    :param frame: frame number
    :return: sets of coordinates values
    """
    coordinates = []
    for i in range(len(a)):
        coordinates.append(df[a[i]][frame])
    return coordinates


def compute_angle(coordinates):
    """
    Computes the angle in degrees given a set of (3,2) or (3, 3) coordinates.
    :param coordinates:
    :return:
    """
    if coordinates.shape == (3, 3):
        return get_angle_3d(coordinates)
    if coordinates.shape == (3, 2):
        return get_angle_2d(coordinates)
    else:
        raise ("The coordinates set have different length, it can only compute an angle in 3d space or 2d not mixed. "
               "\nThis could happen if a marker has xy coordinate and another one xyz."
               "\n Or if your markers are named ambiguously and multiple ones have the same name.")
        return


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
