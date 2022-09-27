import c3d
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
        labels = get_c3d_labels(f)
        labels.insert(0, "frame_n")

        run = []
        for frame_no, points, analog in c3d.Reader(f).read_frames(copy=False):
            fields = [frame_no]
            for x, y, z, err, cam in points:
                fields.append(x)
                fields.append(y)
                fields.append(z)
            run.append(fields)

    markers_df = pd.DataFrame(run, columns=labels)
    return markers_df


def get_c3d_labels(handle):
    reader = c3d.Reader(handle)
    a = reader._groups["POINT"]._params["LABELS"]
    C, R = a.dimensions
    labels = [a.bytes[r * C: (r + 1) * C].strip().decode().lower() for r in range(R)]
    labels_3d = [i + c for i in labels for c in ("_x", "_y", "_z")]
    return labels_3d
