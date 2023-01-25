# neurokin

A Python package to support analysis of neural and kinematic data. 
neurokin supports open_ephys and TDT formats natively for neural and .c3d files for kinematics data but can be expanded easily. Please get in touch if you have another type of data.

Here is how to import neural data (scroll down for the kinematics data):

```python
from neurokin import NeuralData

cool_neural = NeuralData(path="path_to_my_cool_data")
cool_neural.load_tdt_data(stream_name="NPr1", sync_present=True, stim_stream_name="Wav1")

```

Then you can access your cool data and its attributes with ease

```python

# Peek in the raw data
channel_of_interest = 2
cool_neural.raw[channel_of_interest]

# Get the sampling frequency
cool_neural.fs

# Have a look at the sync/stim data
cool_neural.sync_data

# If you have multiple sync channels and you want to pick one you can use

cool_neural.pick_sync_data(0)
# to set it to a specific one
#or simply pick it on the fly with

cool_neural.sync_data(1)

```

#### Now you can access all the important info in an easy way. Typically you need: sampling frequency, raw data, stimulation channel. You can access all of them as _attributes_ of the _object_ condition_x (which is an _instance_ of the class NeuralData)

```python
cool_neural.fs # sampling frequency
cool_neural.raw # raw data
cool_neural.sync_data # stimulation data
```


#### You can also plot a spectrogram with

```python

neural_plot.plot_spectrogram(ax=ax[0],                 # on which ax to plot
                             fs=cool_neural.fs,        # sampling freq
                             raw=cool_neural.raw[0],   # channel to plot
                             ylim=[0, 300],            # frequencies of interest
                             title="ch "+str(rch),     # title
                             nfft=NFFT,                # n of points for the fft
                             noverlap=NOV)             # overlap for fft
```

##### Or a PSD
```python
freq, pxx = processing.calculate_power_spectral_density(cool_neural.raw[0], cool_neural.fs, nperseg=NFFT, noverlap=NOV,
                                                        scaling="spectrum")
```

Other things you can do on the neural side (and I promise to update the readme soon):
- get the timestamps of the stimulation, given the stimulation channel
- parse the raw signal accordingly
- average the signal between pulses
