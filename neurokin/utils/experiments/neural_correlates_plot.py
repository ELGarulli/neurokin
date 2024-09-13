import numpy as np
from neurokin.utils.experiments.neural_states_helper import compute_ci


def plot_psd_single_state(ax, df, group, condition, state, freqs, color, idx_min, idx_max, stat="std", linewidth=3):
    """
    Plots the Power Spectral Denisty trace of a single state, on a given ax.

    :param ax: axis to plot on
    :param group: which subject group to plot
    :param condition: which experimental condition to plot
    :param state: which state to plot
    :param freqs: frequencies corresponding to the full array of PSD values
    :param idx_min: index of the minimum frequency to plot
    :param idx_max: index of the maximum frequency to plot
    :param color: color of the trace
    :param stat: which stat to use to compute the shading, default standard deviation
    :param linewidth: how thick to draw the mean line.
    :return:
    """

    f = freqs[idx_min:idx_max]
    data = df[(df["group"] == group) & (df["condition"] == condition)][state]
    data_mean = data.mean(axis=0)[idx_min:idx_max]
    ax.plot(f, data_mean, color=color, linewidth=linewidth)
    add_shades(ax, f, data, idx_min, idx_max, color, stat=stat)


def add_shades(ax, f, df, idx_min, idx_max, color, stat="std"):
    """
    Helper function to plot shades.

    :param ax: the ax to plot on
    :param f: frequencies to use on the x-axis
    :param df: the data to be plot (array like of states)
    :param idx_min: minimum frequency index to plot
    :param idx_max: maximum frequency index to plot
    :param color: color to plot the shade in, with 0.3 alpha factor
    :param stat: which statistics to use for shades. "std" for standard deviation,
                "sem" for standard error of mean, "ci" for confidence interval
    :return:
    """
    data = df.dropna().values
    if stat == "sem":
        s = data.sem(axis=0)
        s = data.std(axis=0, ddof=1) / np.sqrt(np.size(data))
    elif stat == "ci":
        s = data.apply(compute_ci, axis=0)
        s = compute_ci(data)
    elif stat == "std":
        s = data.std(axis=0)
    else:
        raise ValueError(f"The statistic value {stat} is unsupported, please choose between std, sem or ci")

    lower = data.mean(axis=0) - s
    upper = data.mean(axis=0) + s
    ax.fill_between(f, lower[idx_min:idx_max], upper[idx_min:idx_max], alpha=0.3, color=color)
