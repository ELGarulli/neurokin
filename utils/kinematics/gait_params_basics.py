import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
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


def get_phase_at_max_amplitude(input_signal, fs):
    #freq, pxx = signal.welch(input_signal, fs)
    # freq = np.fft.fftfreq(signal.shape[-1], 1/fs)
    #TODO not working find way to get the frequency at max amplitude and the the complex value
    pxx = np.fft.fft(input_signal)
    freq_of_interest = np.argmax(pxx.real)
    phase = np.angle(pxx[freq_of_interest])
    return phase


def compare_phase(a, b, fs):
    phase_a = get_phase_at_max_amplitude(a, fs)
    phase_b = get_phase_at_max_amplitude(b, fs)
    return phase_a - phase_b
