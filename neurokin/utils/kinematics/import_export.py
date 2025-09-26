import itertools
import c3d
import numpy as np
import pandas as pd


def import_c3d(path):
    """
    Fills df from c3d file

    :param path: path to .c3d
    :return: first_frame, last_frame, sample_rate, df
    """
    with open(path, "rb") as f:
        c3d_reader = c3d.Reader(f)
        first_frame = c3d_reader._header.first_frame
        last_frame = c3d_reader._header.last_frame
        sample_rate = c3d_reader._header.frame_rate
        bodyparts = get_c3d_labels(f)
        axis = ["x", "y", "z"]
        run = []
        for frame_no, points, analog in c3d_reader.read_frames(copy=False):
            fields = []
            for x, y, z, err, cam in points:
                fields.append(x)
                fields.append(y)
                fields.append(z)
            run.append(fields)
        run = np.asarray(run)
        columns = [ f"{b}_{a}" for b, a in itertools.product(bodyparts, axis)]
        df = pd.DataFrame(run, columns=columns)

    return first_frame, last_frame, sample_rate, df


#TESTME
def create_empty_df(scorer, bodyparts, frames_no):
    """
    Creates empty dataframe to receive 3d data frm c3d file

    :param scorer: mock data scorer
    :param bodyparts: list of bodyparts that will be in the dataframe
    :param frames_no: number of frames 
    :return: empty dataframe with correct shape and columns
    """
    
    dataFrame = None
    a = np.full((frames_no, 3), np.nan)
    for bodypart in bodyparts:
        pdindex = pd.MultiIndex.from_product(
            [[scorer], [bodypart], ["x", "y", "z"]],
            names=["scorer", "bodyparts", "coords"])

        frame = pd.DataFrame(a, columns=pdindex, index=range(0, frames_no))
        dataFrame = pd.concat([frame, dataFrame], axis=1)
    return dataFrame


def get_c3d_labels(handle):
    """
    Reads in the labels from a .c3d handle

    :param handle:
    :return: labels
    """
    reader = c3d.Reader(handle)
    a = reader._groups["POINT"]._params["LABELS"]
    C, R = a.dimensions
    labels = [a.bytes[r * C: (r + 1) * C].strip().decode() for r in range(R)]
    return labels


def import_dlc_df(filename):
    if filename.endswith(".csv"):
        df = pd.read_csv(filename, header=[0, 1, 2], skipinitialspace=False, index_col=[0])
    elif filename.endswith(".hd5"):
        df = pd.read_hdf(filename)
    else:
        raise ValueError("File format not supported, please provide .csv or .hd5 files for DLC datasets")
    return df
