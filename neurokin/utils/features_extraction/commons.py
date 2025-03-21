import numpy as np
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def compute_angle(vectors):
    """
    Computes the angle between 3 points in a 3d or 2d space.
    :param vectors: input coordinates
    """
    vectors = np.asarray(vectors)
    if vectors.ndim == 1:
        if vectors.shape[0] in (6, 9):
            vectors = vectors.reshape(1, -1)
        else:
            raise IndexError(f"Expected 1D array of length 6 or 9, but got length {vectors.shape[0]}.")

    try:
        if vectors.shape[1] == 9:
            a, b, c = vectors[:, :3], vectors[:, 3:6], vectors[:, 6:9]
        elif vectors.shape[1] == 6:
            a, b, c = vectors[:, :2], vectors[:, 2:4], vectors[:, 4:6]
    except IndexError:
        raise IndexError("The angles can only be computed on 3 points in a 3d or 2d space. The vectors shape should"
                         f"be (n, 6) or (n, 9) but got shape {vectors.shape}")

    bas = b - a
    bcs = c - b
    angles = []
    for ba, bc in zip(bas, bcs):
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        a = np.arccos(cosine_angle)
        angles.append(a)
    return np.array(angles)


def compute_angle_correlation(vectors):
    """
    Computes the correlation between the angles of a set of vectors.
    :param vectors: input coordinates
    """
    angles = compute_angle(vectors)
    angle_corr = np.corrcoef(angles)
    return angle_corr


def compute_angle_velocity(vectors):
    """
    Computes the velocity of the angle of a set of vectors.
    :param vectors: input coordinates
    """
    angles = compute_angle(vectors)
    angle_vel = np.gradient(angles, 1)
    return angle_vel


def compute_angle_acceleration(vectors):
    """
    Computes the acceleration of the angle of a set of vectors.
    :param vectors: input coordinates
    """
    angle_acc = np.gradient(compute_angle_velocity(vectors), 1)
    return angle_acc


def compute_angle_phase(vectors):
    """
    Computes the phase of a signal.
    First computes the fft, then the phase from the complex element at each index.
    :param vectors: input coordinates
    :return: phases
    """
    angles = compute_angle(vectors)
    spectrum = np.fft.fft(angles)
    phase = np.angle(spectrum)
    return phase


def compute_phase_at_max_amplitude(vectors):
    """
    Computes the phase of a signal at frequency that has the maximum amplitude.
    First computes the fft, then gets the index of maximum value from the real component, then the phase from the
    complex element at that index.

    :param vectors: input coordinates
    :return: phase
    """
    angles = compute_angle(vectors)
    c_transform = np.fft.fft(angles)
    r_transform = abs(c_transform) ** 2
    freq_of_interest = np.argmax(r_transform)
    phase = np.angle(c_transform[freq_of_interest], deg=True)
    phase = phase if phase > 0 else 360 + phase
    return phase


def compute_speed(df: pd.DataFrame) -> NDArray:
    """
    Computes the speed of a trajectory.
    :param df: input coordinates
    """
    traj = df.apply(compute_velocity)
    speed = np.apply_along_axis(np.linalg.norm, 1, traj, None, 0)
    return speed


def compute_velocity(df: pd.DataFrame) -> NDArray:
    """
    Computes the velocity of a trajectory.
    :param df: input coordinates
    """
    df = df.values
    return np.gradient(df, 1)


def compute_acceleration(df: pd.DataFrame) -> NDArray:
    """
    Computes the acceleration of a trajectory.
    :param df: input coordinates
    """
    velocity = compute_velocity(df)
    return np.gradient(velocity, 1)


def compute_tang_acceleration(df: pd.DataFrame) -> NDArray:
    """
    Computes the tangential acceleration of a trajectory.
    :param df: input coordinates
    """
    speed = compute_speed(df)
    return np.gradient(speed, 1)
