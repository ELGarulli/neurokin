from array import array
import numpy as np

def export_neural_data_to_bin(data, filename):
    """
    Export the raw data to a binary file in C major order.

    :param data: data to be stored, should have shape n_ch*n_sample
    :param filename: name of the file
    :return:
    """
    data_flat = data.flatten(order='C')
    data_flat.tofile(filename)
