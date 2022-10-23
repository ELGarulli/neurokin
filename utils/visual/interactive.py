from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

def plot_sliding(array):
    fig, ax = plt.subplots()
    line, = ax.plot(array)
    fig.subplots_adjust(bottom=0.25)
    fig.subplots_adjust(left=0.25, bottom=0.25)
    axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    freq_slider = Slider(
        ax=axfreq,
        label='Time [samples]',
        valmin=0.0,
        valmax=len(array),
        valinit=0,
    )
    def update(val):
        pos = int(freq_slider.val)
        ax.axis([pos, pos+10000, min(array[pos:pos+10000]), max(array[pos:pos+10000])])
        fig.canvas.draw_idle()
    freq_slider.on_changed(update)
    # Display the plot
    plt.show()