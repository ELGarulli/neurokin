import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def spider_factory(num_vars, frame='circle'):
    """
    Taken from https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def get_polygon_corners(theta, bottom, top):
    """
    Retrieves the corners of a polygon encompassed by the shades borders to allow for proper shading

    :param theta: radius reference
    :param bottom: lower border of the shading. y values
    :param top: upper border of the shading. y values
    :return: x and y values of the polygon encompassed by the borders
    """
    x = []
    y = []
    for i in theta:
        x.append(i)
        x.append(i)
    for i in range(len(theta)):
        y.append(bottom[i])
        y.append(top[i])
    return x, y


def plot_spider_single_trace(ax, data, color, theta, zorder=0):
    """
    Plots a single trace on a spider plot (radar plot)

    :param ax: axis to plot on
    :param data: dataset to plot
    :param color: color to use to plot
    :param theta: radius reference
    :param zorder: z-order for stacking
    :return:
    """
    count_avg = data["mean"].values
    n_stat = data["lower_bound"].values
    n_stat[n_stat < 0] = 0
    p_stat = data["upper_bound"].values

    ax.plot(theta, count_avg, color=color, linewidth=3, zorder=zorder + 10)

    ax.scatter(theta, count_avg, color=color, s=50, zorder=zorder + 20)

    ax.fill_between(theta, n_stat, p_stat,
                    facecolor=color, alpha=0.4, zorder=zorder)

    ax.fill([theta[0], theta[0], theta[-1], theta[-1]], [p_stat[0], n_stat[0], n_stat[-1], p_stat[-1]],
            facecolor=color, alpha=0.4, zorder=zorder)
