from array import array
import numpy as np

def export_neural_data_to_bin(data, filename):
    #TODO this doesnt work
    data_flat = data.flatten(order='C')
    data_flat.astype('int16').tofile(filename)
    print("No no no no no, this function is not good. I still have to re-write it. Don't trust it.")