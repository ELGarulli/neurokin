import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

#TESTME
def get_angle(coordinates):
    """
    Get angle between 3 points in a 3d or 2d space
    :param coordinates: array with coordinates of 3 points
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

#TESTME
def get_phase_at_max_amplitude(input_signal):
    """
    Computes the phase of a signal at frequency that has the maximum amplitude.
    First computes the fft, then gets the index of maximum value from the real component, then the phase from the
    complex element at that index.
    :param input_signal:
    :return: phase
    """
    c_transform = np.fft.fft(input_signal)
    r_transform = abs(c_transform) ** 2
    freq_of_interest = np.argmax(r_transform)
    phase = np.angle(c_transform[freq_of_interest], deg=True)
    phase = phase if phase > 0 else 360 + phase
    return phase

#TESTME
def get_phase(input_signal):
    """
    Computes the phase of a signal.
    First computes the fft, then the phase from the complex element at each index.
    :param input_signal:
    :return: phases
    """
    spectrum = np.fft.fft(input_signal)
    phase = np.angle(spectrum)
    return phase

#TESTME
def compare_phase(signal_a, signal_b):
    """
    Gets the difference of phase of two signals

    :param signal_a: input signal
    :param signal_b: input signal to compare
    :return: phase of signal_a - phase of signal_b
    """
    phase_a = get_phase_at_max_amplitude(signal_a)
    phase_b = get_phase_at_max_amplitude(signal_b)
    return phase_a - phase_b
