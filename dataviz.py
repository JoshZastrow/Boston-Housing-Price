# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 08:33:33 2018

@author: joshua
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import animation
from keras.datasets import boston_housing
import matplotlib.patches as patches
import matplotlib.path as path
from matplotlib import rc

rc('animation', html='html5')  # sets animation display from none to html5

class graph_animator(animation.TimedAnimation):

    def __init__(self):
        self.fig = plt.figure(figsize=(9,9))
        self.fig.suptitle('Are Any Features Closely Correlated?',
                          x=0.4, y=0.85,
                          horizontalalignment='center',
                          fontsize=16,
                          fontweight='bold')
        # Subplot layout
        self.gs = gridspec.GridSpec(6,6)
        self.gs.update(left=0.14,
                       right=0.99,
                       wspace=0.2,
                       hspace=.1,
                       top=0.95,
                       bottom=0.1)

        self.ax1 = self.fig.add_subplot(self.gs[2:6, 0:4])
        self.ax2 = self.fig.add_subplot(self.gs[1:2, 0:4])
        self.ax3 = self.fig.add_subplot(self.gs[2:6, 4:5])

        self.ax1.spines['bottom'].set_color('#666B73')
        self.ax1.spines['top'].set_color('white')
        self.ax1.spines['right'].set_color('white')
        self.ax1.spines['left'].set_color('#666B73')

        self.ax2.axes.set_axis_off()
        self.ax3.axes.set_axis_off()

        # Headers
        self.headers = ["Per Capita Crime",
                        "Zoned over 25k sq-ft",
                        "Non-retail Acres Per Town",
                        "On the Charles?",
                        "NO2 Levels ppm",
                        "Ave Number of Rooms",
                        "Portion 40+ y.o Houses",
                        "Distance to City",
                        "Highway Accesibility",
                        "Property Tax Rate",
                        "Pupil-Teacher Ratio",
                        "Portion of African-American",
                        "Percent Lower Status"]

        (x_train, _), _ = boston_housing.load_data()

        self.data_series = self.create_transform(x_train,
                                                 time_steps=20,
                                                 delay=80)

        self._create_histograms()

        # Scatter plot information
        self.x1_ttl = self.ax1.text(.5, -0.1, '',
                                    horizontalalignment='center',
                                    transform=self.ax1.transAxes,
                                    fontweight='bold', fontsize=12)
        self.y1_ttl = self.ax1.text(-0.1, .5, '',
                                    transform=self.ax1.transAxes,
                                    horizontalalignment='left',
                                    verticalalignment='center',
                                    rotation=90,
                                    fontweight='bold',
                                    fontsize=12)

        self.line, = self.ax1.plot(self.data_series[0][2],
                                   self.data_series[0][3],
                                   'o', c='black',
                                   markerfacecolor= '#FEEAA8',
                                   linewidth=3,
                                   markersize=8)

        animation.TimedAnimation.__init__(self, self.fig, interval=5, blit=True)
        self._drawn_artists = []

    def _draw_frame(self, framedata):

        i = framedata
        x_title, y_title, x_data, y_data = self.data_series[i]
        self.line.set_xdata(x_data)
        self.line.set_ydata(y_data)

        self.ax1.set_xlim(x_data.min(), x_data.max())
        self.ax1.set_ylim(y_data.min(), y_data.max())

        self.x1_ttl.set_text(x_title)
        self.y1_ttl.set_text(y_title)
        xn, _ = np.histogram(x_data, bins=self.nbins)
        yn, _ = np.histogram(y_data, bins=self.nbins)

        bottom = np.zeros(self.nbins)
        x_top = bottom + xn  # freq[0]
        y_top = bottom + yn  # freq[1]

        self.verts[0, 1::5, 1] = x_top
        self.verts[1, 1::5, 0] = y_top
        self.verts[0, 2::5, 1] = x_top
        self.verts[1, 2::5, 0] = y_top

        self.ax2.set_ylim(bottom.min(), x_top.max())
        self.ax3.set_xlim(bottom.min(), y_top.max())

        self._drawn_artists = [self.line,
                               self.x1_ttl, self.y1_ttl,
                               self.x_patch, self.y_patch,
                               ]

    def new_frame_seq(self):
        return iter(range(len(self.data_series)))

    def _init_draw(self):
        """Clears the axis"""
        lines = [self.line]
        for l in lines:
            l.set_data([], [])

    def _create_histograms(self):

        # Histograms
        self.nbins = 20  # unmber of bins
        xn, xbins = np.histogram(self.data_series[0][2], bins=self.nbins)
        yn, ybins = np.histogram(self.data_series[0][3], bins=self.nbins)

        # get edges of histogram bars
        x_left = np.array(xbins[:-1])
        y_left = np.array(ybins[:-1])
        x_right = np.array(xbins[:-1])
        y_right = np.array(ybins[:-1])
        x_bottom = np.zeros(self.nbins)
        y_bottom = np.zeros(self.nbins)
        x_top = xn
        y_top = yn

        num_verts = self.nbins * (1 + 3 + 1) # 1 move to, 3 vertices, 1 close poly
        self.verts = np.zeros(shape=(2, num_verts, 2))  # (axis, value, coordinate)

        # x axis
        self.verts[0, 0::5, 0] = x_left
        self.verts[0, 0::5, 1] = x_bottom
        self.verts[0, 1::5, 0] = x_left
        self.verts[0, 1::5, 1] = x_top
        self.verts[0, 2::5, 0] = x_right
        self.verts[0, 2::5, 1] = x_top
        self.verts[0, 3::5, 0] = x_right
        self.verts[0, 3::5, 1] = x_bottom

        # y axis
        self.verts[1, 0::5, 0] = y_bottom
        self.verts[1, 0::5, 1] = y_left
        self.verts[1, 1::5, 0] = y_top
        self.verts[1, 1::5, 1] = y_left
        self.verts[1, 2::5, 0] = y_top
        self.verts[1, 2::5, 1] = y_right
        self.verts[1, 3::5, 0] = y_bottom
        self.verts[1, 3::5, 1] = y_right

        # Drawing Codes
        codes = np.ones((num_verts), int) * path.Path.LINETO  # Instructions
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY

        x_path = path.Path(self.verts[0], codes)
        y_path = path.Path(self.verts[1], codes)

        self.x_patch = patches.PathPatch(x_path,
                                         facecolor='#FA6367',
                                         edgecolor='#78C9EC',
                                         linewidth=15,
                                         alpha=1)
        self.y_patch = patches.PathPatch(y_path,
                                         facecolor='#FA6367',
                                         edgecolor='#78C9EC',
                                         linewidth=15,
                                         alpha=1)

        self.ax2.add_patch(self.x_patch)
        self.ax3.add_patch(self.y_patch)

        self.ax2.set_xlim(xbins[0], xbins[-1])
        self.ax3.set_ylim(ybins[0], ybins[-1])
        self.ax2.set_ylim(x_bottom.min(), x_top.max())
        self.ax3.set_xlim(y_bottom.min(), y_top.max())


    def create_transform(self, data, time_steps=60, delay=40):
        frame_data = []
        rows, cols = data.shape

        curr_i = 0
        curr_j = 0
        next_i = 1
        next_j = 1

        while next_i < cols:
            next_j = next_i # don't repeat previous comparisons
            while next_j < cols:

                x_title = self.headers[curr_i]
                y_title = self.headers[curr_j]
                x_data = data[:, curr_i]

                # Create the transitioning y data
                y_data = np.array([np.linspace(data[r, curr_j],
                                                    data[r, next_j],
                                                    time_steps) for r in range(rows)])

                # Create a list of frames for the transition
                for t in range(delay):
                    frame_data += [(x_title, y_title, x_data, data[:, curr_j])]

                for t in range(time_steps):
                    frame_data += [(x_title, y_title, x_data, y_data[:, t])]

                curr_j = next_j
                next_j += 1

            # Create transitioning x data
            x_data = np.array([np.linspace(data[r, curr_i],
                                           data[r, next_i],
                                           time_steps) for r in range(rows)])
            # Add Frames
            for t in range(time_steps):
                frame_data += [(x_title, y_title, x_data[:, t], y_data[:, -1])]

            curr_i = next_i
            next_i += 1

        return frame_data


class plot_handler():
    """
    ' Plot handler to help me control the shape of my subplots better.
    """
    def __init__(self, plot_rows, plot_cols):
        self.rows = plot_rows
        self.cols = plot_cols
        self.fig = plt.figure(facecolor='white', figsize=(16,16))
        self.grid = gridspec.GridSpec(self.rows, self.cols)

        self.grid.update(left=0.1,
                         right=0.9,
                         wspace=0.2,
                         hspace=.1,
                         top=0.9,
                         bottom=0.1)

        self.ax = {}
        self.xlimit = None
        self.ylimit = None

    def add_plot(self, top, bottom, left, right, name):
        self.ax[name] = self.fig.add_subplot(self.grid[top:bottom, left:right])
        self.ax[name].set_title(name, fontweight="bold", size=14)

    def plot_exists(self, name):
        return name in self.ax

    def plot(self, data, plot_name, data_name, ylim=None):
        self.ax[plot_name].plot(data,  '-', label=data_name, animated=True)

        if not ylim:
            self.ax[plot_name].set_ylim([0,ylim])



def _blit_draw(self, artists, bg_cache):
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = []
    for a in artists:
        # If we haven't cached the background for this axes object, do
        # so now. This might not always be reliable, but it's an attempt
        # to automate the process.
        if a.axes not in bg_cache:
            # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
            # change here
            bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
        a.axes.draw_artist(a)
        updated_ax.append(a.axes)

    # After rendering all the needed artists, blit each axes individually.
    for ax in set(updated_ax):
        # and here
        # ax.figure.canvas.blit(ax.bbox)
        ax.figure.canvas.blit(ax.figure.bbox)


if __name__ == "__main__":
#
#    # MONKEY PATCH!!
#    animation.Animation._blit_draw = _blit_draw

    headers = ["Per Capita Crime",
               "Zoned over 25k sq-ft",
               "Non-retail Acres Per Town",
               "On the Charles?",
               "NO2 Levels ppm",
               "Ave Number of Rooms",
               "Portion 40+ y.o Houses",
               "Distance to City",
               "Highway Accesibility",
               "Property Tax Rate",
               "Pupil-Teacher Ratio",
               "Portion of African-American",
               "Percent Lower Status"]

    ani = graph_animator()

    plt.show()
