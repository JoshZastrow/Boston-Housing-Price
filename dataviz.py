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
    
    last_x = None
    last_y = None
    
    for i, x in enumerate(headers):     
        for j, y in enumerate(reversed(headers)):

            if not last_x: 
                last_x = x
                last_i = i
            
            if not last_y: 
                last_y = y
                last_j = j
                
            x_title = last_x
            y_title = last_y
            x_data = data[:, last_i]
            
            # Create the transition data between each column
            y_data = np.array([np.linspace(data[r, last_j], 
                                                data[r, j], 
                                                time_steps) for r in range(rows)])
    
            # Create a list of frames for the transition
            for t in range(delay):
                frame_data += [(x_title, y_title, x_data, data[:, last_j])]
                
            for t in range(time_steps):
                frame_data += [(x_title, y_title, x_data, y_data[:, t])]
                
            last_y = y
            last_j = j
        
        last_x = x
        last_i = i
        
    return frame_data
            
def data_feed(data):
    while True:
        yield data.pop(0)
        
def animate(data_packet):
    x_title, y_title, x_data, y_data = data_packet
    
    line.set_xdata(x_data)
    line.set_ydata(y_data)
    
    ax.set_xlim(x_data.min(), x_data.max())
    ax.set_ylim(y_data.min(), y_data.max())
    
    xttl.set_text(x_title)
    yttl.set_text(y_title)
    
    return line, xttl, yttl

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
#    matplotlib.animation.Animation._blit_draw = _blit_draw
    
    headers = ["CRIM", "ZN","INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", 
                "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    x_train = p.StandardScaler().fit_transform(x_train)
    data = create_transform(x_train, 
                            y_train, 
                            time_steps=30, 
                            delay=50)
    
    fig = plt.figure(figsize=(8,8))
    
    gs = gridspec.GridSpec(5,5)
    gs.update(left=0.1,  
              right=0.8, 
              wspace=0.2,
              hspace=.1,
              top=0.8, 
              bottom=0.1)
#    
    ax = fig.add_subplot(gs[1:5, 0:4])
#    
#    ax2 = fig.add_subplot(gs[0:1,0:4])
    ax.set_xticks([])
    ax.set_yticks([])
#    ax2.set_title('Tester')
#    
#    ax3 = fig.add_subplot(gs[1:5, 4:5])
#    ax3.set_xticks([])
#    ax3.set_yticks([])
#    ax3.set_title('tester2')
#    
    line, = ax.plot(data[0][2], data[0][3], 'o')
#    hist, = ax2.plot()
    x_max = np.max(8)
    x_min = np.min(-8)
    xttl = ax.text(.5, -0.1, 'test', 
                   transform = ax.transAxes, 
                   fontweight='bold', fontsize=12)
    yttl = ax.text(-.10, .5, 'test2', 
                   transform = ax.transAxes, 
                   rotation=90, 
                   fontweight='bold', fontsize=12)
    
    anim = animation.FuncAnimation(fig, 
                                   animate,
                                   frames=data_feed(data),blit=True,
                                   interval=10)

#    anim.save('test_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
