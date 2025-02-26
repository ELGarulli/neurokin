import numpy as np
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def angle(vectors):
    if vectors.shape[1] == 9:
        a, b, c = vectors[:, :3], vectors[:, 3:6], vectors[:, 6:9]
    elif vectors.shape[1] == 6:
        a, b, c = vectors[:, 2], vectors[:, 2:4], vectors[:, 4:6]

    bas = a - b
    bcs = c - b
    angles = []
    for ba, bc in zip(bas, bcs):
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        angles.append(angle)
    return np.array(angles)


def compute_speed(df: pd.DataFrame, fs: float) -> NDArray:
    traj = df.apply(compute_velocity, args=(fs,))
    speed = np.apply_along_axis(np.linalg.norm, 1, traj, None, 0)
    return speed


def compute_velocity(df: pd.DataFrame, fs: float) -> NDArray:
    df = df.values
    return np.gradient(df, (1 / fs))


def compute_acceleration(df: pd.DataFrame, fs: float) -> NDArray:
    velocity = compute_velocity(df, fs)
    return np.gradient(velocity, (1 / fs))


def compute_tang_acceleration(df: pd.DataFrame, fs: float) -> NDArray:
    speed = compute_speed(df, fs)
    return np.gradient(speed, (1 / fs))
