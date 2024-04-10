import c3d
import numpy as np
import pandas as pd


#TESTME
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
        scorer = "scorer"
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

        df = create_empty_df(scorer, bodyparts, np.shape(run)[0])
        count = 0
        for bodypart in bodyparts:
            for a in axis:
                df[scorer, bodypart, a] = run[:, count]
                count += 1

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


#TESTME
def get_c3d_labels(handle):
    """
    Reads in the labels from a .c3d handle
    :param handle:
    :return: labels
    """
    reader = c3d.Reader(handle)
    a = reader._groups["POINT"]._params["LABELS"]
    C, R = a.dimensions
    labels = [a.bytes[r * C: (r + 1) * C].strip().decode().lower() for r in range(R)]
    return labels