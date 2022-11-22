import c3d
import numpy as np
import pandas as pd


def c3d2csv(filename):
    with open(filename, "rb") as f:
        labels = get_c3d_labels(f)
        labels.insert(0, "frame_n")

        run = []
        for frame_no, points, analog in c3d.Reader(f).read_frames(copy=False):
            fields = [frame_no]
            for x, y, z, err, cam in points:
                fields.append(str(x))
                fields.append(str(y))
                fields.append(str(z))
            run.append(fields)

    markers_df = pd.DataFrame(run, columns=labels)
    markers_df.to_csv(filename.replace('.c3d', '.csv'), sep="\t")


def import_c3d(filename):
    with open(filename, "rb") as f:
        c3d_reader = c3d.Reader(f)
        first_frame = c3d_reader._header.first_frame
        last_frame = c3d_reader._header.last_frame
        frame_rate = c3d_reader._header.frame_rate
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

        multiindex_df = create_empty_df(scorer, bodyparts, np.shape(run)[0])
        count = 0
        for bodypart in bodyparts:
            for a in axis:
                multiindex_df[scorer, bodypart, a] = run[:, count]
                count += 1

    return first_frame, last_frame, frame_rate, multiindex_df


def create_empty_df(scorer, bodyparts, frames_no):
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
    reader = c3d.Reader(handle)
    a = reader._groups["POINT"]._params["LABELS"]
    C, R = a.dimensions
    labels = [a.bytes[r * C: (r + 1) * C].strip().decode().lower() for r in range(R)]
    return labels
