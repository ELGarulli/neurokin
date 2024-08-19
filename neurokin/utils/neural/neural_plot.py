import numpy as np
from numpy.typing import ArrayLike
from typing import List
from matplotlib import axes as ax


def plot_spectrogram(ax: ax, fs: float, raw: ArrayLike, ylim: List[int], title: str = "", tick_spacing: int = 10,
                     nfft: int = None, noverlap: int = None, vmin=None, vmax=None, **kwargs) -> ax:
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
    ax.set_title(title)
    ax.specgram(raw, NFFT=nfft, Fs=fs, noverlap=noverlap, vmin=vmin, vmax=vmax, **kwargs)
    ax.set_ylim(ylim)
    ax.set_yticks(np.arange(ylim[0], ylim[1], tick_spacing))
    return ax


def plot_welch(ax: ax, freqs: ArrayLike, pxx_den: ArrayLike, title: str = "PSD", xlim=None,
               ylim=None,
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
    ax.set_title(title)

    if not xlim is None:
        ax.set_xlim(xlim)
        ticks_min = xlim[0]
        ticks_max = xlim[1]
    else:
        ticks_min = min(freqs)
        ticks_max = max(freqs)

    if not ylim is None:
        ax.set_ylim(ylim)

    ax.set_xticks(np.arange(ticks_min, ticks_max, tick_spacing))

    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('PSD [V**2/Hz]')

    ax.semilogy(freqs, pxx_den)
    return ax
