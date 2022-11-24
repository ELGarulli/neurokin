# neurokin

A Python package to support analysis of neural and kinematic data.

Here is how to import neural data:

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

Other things you can do on the neural side (and I promise to update the readme soon):
- get the timestamps of the stimulation, given the stimulation channel
- parse the raw signal accordingly
- average the signal between pulses
