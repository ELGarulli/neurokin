import numpy as np
from numpy.typing import ArrayLike
from typing import List
from matplotlib import axes as ax


def plot_spectrogram(ax: ax, fs: float, raw: ArrayLike, title: str, ylim: List[int, int], tick_spacing: int = 10,
                     nfft: int = None, noverlap: int = None) -> ax:
    """
    Wrapper for matplotlib spectrogram to add some style elements.
    :param ax: axes where to plot
    :param fs: sampling frequency
    :param raw: raw data
    :param title: title of the figure
    :param ylim: limit of frequencies
    :param tick_spacing: spacing of the ticks on the axis
    :param nfft: nfft to comput the fft
    :param noverlap: number of overlap points
    :return:
    """
    ax.title(title)
    ax.specgram(raw, NFFT=nfft, Fs=fs, noverlap=noverlap)
    ax.ylim(ylim)
    ax.yticks(np.arange(ylim[0], ylim[1], tick_spacing))
    return ax


def plot_welch(ax: ax, freqs: ArrayLike, pxx_den: ArrayLike, title: str, xlim: List[int, int], ylim: List[int, int],
               tick_spacing: int = 10) -> ax:
    """
    Returns a plot of psd with some style elements
    :param ax: axes where to plot
    :param freqs: array like of frequencies corresponding to the x axis
    :param pxx_den: array like of powers corresponding to the y axis
    :param title: str
    :param xlim: limit of frequencies for plotting
    :param ylim: limit of power for plotting
    :param tick_spacing:  spacing of the ticks on the axis
    :return:
    """
    ax.title(title)
    ax.xlim(xlim)
    ax.xticks(np.arange(xlim[0], xlim[1], tick_spacing))
    ax.ylim(ylim[0], ylim[1])
    ax.xlabel('frequency [Hz]')
    ax.ylabel('PSD [V**2/Hz]')

    ax.semilogy(freqs, pxx_den)
    return ax
