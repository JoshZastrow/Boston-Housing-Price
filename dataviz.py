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
from sklearn import preprocessing as p
import matplotlib.patches as patches
import matplotlib.path as path
from matplotlib import rc
from matplotlib.font_manager import FontProperties

rc('animation', html='html5')  # sets animation display from none to html5

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
        
    def __call__(self):
        plt.show
        
    def add_plot(self, top, bottom, left, right, name):
        self.ax[name] = self.fig.add_subplot(self.grid[top:bottom, left:right])
        self.ax[name].set_title(name, fontweight="bold", size=14)
        
    def plot_exists(self, name):
        return name in self.ax
        
    def plot(self, data, plot_name, data_name, ylim=None, animated=False):
        self.ax[plot_name].plot(data,  '-', label=data_name, animated=animated)
        
        if not ylim:
            self.ax[plot_name].set_ylim([0,ylim])
            
       
def create_transform(data, labels, time_steps=20, delay=200):
    frame_data = []  
    rows, cols = data.shape
    
    curr_i = 0
    curr_j = 0
    next_i = 1
    next_j = 1
    
    while next_i < cols:     
        next_j = next_i # don't repeat previous comparisons
        while next_j < cols:

            x_title = headers[curr_i]
            y_title = headers[curr_j]
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
            

def data_feed(data):
    while True:
        yield data.pop(0)
        
def animate(data_packet):
    x_title, y_title, x_data, y_data = data_packet
    
    line.set_xdata(x_data)
    line.set_ydata(y_data)
    
    ax1.set_xlim(x_data.min(), x_data.max())
    ax1.set_ylim(y_data.min(), y_data.max())
    
    x1_ttl.set_text(x_title)
    y1_ttl.set_text(y_title)
    xn, xbins = np.histogram(x_data, bins=nbins)
    yn, ybins = np.histogram(y_data, bins=nbins)

    bottom = np.zeros(nbins)
    x_top = bottom + xn  # freq[0]
    y_top = bottom + yn  # freq[1]
    
    verts[0, 1::5, 1] = x_top
    verts[1, 1::5, 0] = y_top
    verts[0, 2::5, 1] = x_top
    verts[1, 2::5, 0] = y_top
    
    ax2.set_ylim(bottom.min(), x_top.max())
    ax3.set_xlim(bottom.min(), y_top.max())
    
    return line, x1_ttl, y1_ttl, x_patch, y_patch, 

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

    # MONKEY PATCH!!
    animation.Animation._blit_draw = _blit_draw
    
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
    
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    x_train = p.StandardScaler().fit_transform(x_train)
    data_series = create_transform(x_train, 
                                   y_train, 
                                   time_steps=60, 
                                   delay=40)
    
    fig = plt.figure(figsize=(6,6))
    
    gs = gridspec.GridSpec(6,6)
    gs.update(left=0.1,  
              right=0.95, 
              wspace=0.2,
              hspace=.1,
              top=0.95, 
              bottom=0.1)

    # Main Plot
    ax1 = fig.add_subplot(gs[2:6, 0:4])
    ax1.set_xticks([])
    ax1.set_yticks([])
    x1_ttl = ax1.text(.5, -0.1, '', 
                      horizontalalignment='center',
                      transform = ax1.transAxes, 
                      fontweight='bold', fontsize=12)
    y1_ttl = ax1.text(-0.1, .5, '', 
                      transform = ax1.transAxes, 
                      horizontalalignment='left',
                      verticalalignment='center',
                      rotation=90, 
                      fontweight='bold', # bbox=dict(facecolor='red', alpha=0.5),
                      fontsize=12)
    
    ax1.spines['bottom'].set_color('#666B73')
    ax1.spines['top'].set_color('white') 
    ax1.spines['right'].set_color('white')
    ax1.spines['left'].set_color('#666B73')
              
    # Top Histogram
    ax2 = fig.add_subplot(gs[1:2, 0:4])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axes.set_axis_off()
    
    # Right Histogram
    ax3 = fig.add_subplot(gs[2:6, 4:5])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.axes.set_axis_off()
    
    # Histograms
    nbins = 20  # unmber of bins
    xn, xbins = np.histogram(data_series[0][2], bins=nbins)
    yn, ybins = np.histogram(data_series[0][3], bins=nbins)
    
    
    # get edges of histogram bars
    x_left = np.array(xbins[:-1])
    y_left = np.array(ybins[:-1])
    x_right = np.array(xbins[:-1])
    y_right = np.array(ybins[:-1])
    x_bottom = np.zeros(nbins)
    y_bottom = np.zeros(nbins)
    x_top = xn  # freq[0]
    y_top = yn  # freq[1]
 
    num_verts = nbins * (1 + 3 + 1) # 1 move to, 3 vertices, 1 close poly
    verts = np.zeros(shape=(2, num_verts, 2))  # (axis, value, coordinate)
    
    # x axis
    verts[0, 0::5, 0] = x_left
    verts[0, 0::5, 1] = x_bottom
    verts[0, 1::5, 0] = x_left
    verts[0, 1::5, 1] = x_top
    verts[0, 2::5, 0] = x_right
    verts[0, 2::5, 1] = x_top
    verts[0, 3::5, 0] = x_right
    verts[0, 3::5, 1] = x_bottom

    # y axis
    verts[1, 0::5, 0] = y_bottom
    verts[1, 0::5, 1] = y_left
    verts[1, 1::5, 0] = y_top
    verts[1, 1::5, 1] = y_left
    verts[1, 2::5, 0] = y_top
    verts[1, 2::5, 1] = y_right
    verts[1, 3::5, 0] = y_bottom
    verts[1, 3::5, 1] = y_right
    
    # Drawing Codes
    codes = np.ones((num_verts), int) * path.Path.LINETO  # Instructions
    codes[0::5] = path.Path.MOVETO
    codes[4::5] = path.Path.CLOSEPOLY
    
    x_patch = None  # store object later
    y_patch = None
    
    x_path = path.Path(verts[0], codes)
    y_path = path.Path(verts[1], codes)
    
    x_patch = patches.PathPatch(x_path, 
                                facecolor='#FA6367', 
                                edgecolor='#78C9EC', 
                                linewidth=15,
                                alpha=1)
    y_patch = patches.PathPatch(y_path, 
                                facecolor='#FA6367', 
                                edgecolor='#78C9EC',
                                linewidth=15,
                                alpha=1)
    
    ax2.add_patch(x_patch)
    ax3.add_patch(y_patch)
    
    ax2.set_xlim(xbins[0], xbins[-1])
    ax3.set_ylim(ybins[0], ybins[-1])
    ax2.set_ylim(x_bottom.min(), x_top.max())
    ax3.set_xlim(y_bottom.min(), y_top.max())
    
    line, = ax1.plot(data_series[0][2], data_series[0][3], 'o', c='#FA6367',
                     markerfacecolor='#FEEAA8', markersize=5)
    
    font = FontProperties()
    font.set_family('cursive')
    
    fig.suptitle('Multivariate Analysis', x=0.38, y=0.85, 
                 horizontalalignment='center',
                 fontsize=16,
                 fontweight='bold',
                 fontproperties='cursive')
    
    anim = animation.FuncAnimation(fig, 
                                   animate,
                                   frames=data_feed(data_series),
                                   interval=10)
    mywriter = animation.FFMpegWriter()
    anim.save('test_animation.mp4', 
              writer=mywriter, 
              fps=30, 
              extra_args=['-vcodec', 'libx264'])
