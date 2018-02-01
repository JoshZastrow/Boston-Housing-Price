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
        self.ax[name].set_title(name,fontweight="bold", size=14)
        
    def plot_exists(self, name):
        if name in self.ax:
            return True
        else:
            return False
        
    def plot(self, data, plot_name, data_name, ylim=None, animated=False):
        self.ax[plot_name].plot(data,  '-', label=data_name, animated=animated)
        
        if not ylim:
            self.ax[plot_name].set_ylim([0,ylim])
            
            
def create_transform(data, labels, time_steps=20, freeze_steps=200, headers=None):
    """
    take in each column of data and do a linear transform to the next dataset
    """
    # how many frames for animation
    total_frames = data.shape[1] * time_steps 
    total_frames += data.shape[1] * freeze_steps
    total_frames += time_steps + freeze_steps  # add one to transition back to first feature
    # Scale the data
    data = p.StandardScaler().fit_transform(data)
    
    results = np.zeros(shape=(data.shape[0], total_frames + 1))
    
    chart_titles = []
    for t in headers:
        chart_titles += [t] * (time_steps + freeze_steps)
    
    chart_titles += [headers[0]] * (time_steps + freeze_steps)  # return to first column
    
    assert len(chart_titles) == (results.shape[1] - 1)
    
    for i in range(data.shape[1]):
        
        beg = i * time_steps + i * freeze_steps
        mid = i * time_steps + (i + 1) * freeze_steps
        end = (i + 1) * time_steps + (i + 1) * freeze_steps 
        
        # duplicate the actual data for frozen frames
        results[:, beg:mid] = np.array([data[:,i],] * freeze_steps).transpose()
        
        # transform the data from one feature

        for j in range(data.shape[0]):
            if (i + 1) < data.shape[1]:
                linspace = np.linspace(data[j, i], data[j, i + 1], time_steps)
            else:
                linspace = np.linspace(data[j, i], data[j, 0], time_steps)
            results[j, mid:end] = linspace

    
    if np.sum(results[:, -1]) != 0:
        print('Last row has been filled by code... fix')
    else:
        results[:,-1:] = labels.reshape((data.shape[0], 1))
        
    return results, chart_titles
    

def data_feed(data, titles):
    while True:
        for j in range(data.shape[1]):
            t = titles.pop(0)
            yield (data[:,j], t)

def animate(data_packet):
    data, title = data_packet
    line.set_xdata(data)
#    ax2.cla()
#    ax2.set_xlim(x_min, x_max)
#    ax2.set_xticks([])
#    ax2.set_yticks([])
#    ax2.set_title(title)
#    box = ax2.boxplot(x=data, vert=False)
    
    return line, # box,
    
if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    x_headers = ["CRIM", "ZN","INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", 
                 "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    data, chart_titles = create_transform(x_train, 
                                          y_train, 
                                          time_steps=50, 
                                          freeze_steps=50,
                                          headers=x_headers)
    
    fig = plt.figure()
    gs = gridspec.GridSpec(5,2)
    gs.update(left=0.1,  
              right=0.9, 
              wspace=0.2,
              hspace=.1,
              top=0.9, 
              bottom=0.1)
    
    ax = fig.add_subplot(gs[1:5,0:2])
    ax.set_ylabel('Housing Price', fontweight='bold', fontsize=12)
    
#    ax2 = fig.add_subplot(gs[0:1, 0:2])
#    
#    ax2.set_xticks([])
#    ax2.set_yticks([])
#    ax2.set_title('Tester')
    
    line, = ax.plot(data[:,0], data[:,-1:], 'o')
    x_max = np.max(data[:, :-1])
    x_min = np.min(data[:, :-1])
    ax.set_xlim(x_min, x_max)
    
    anim = animation.FuncAnimation(fig, 
                                   animate,
                                   frames=data_feed(data,chart_titles), blit=True,
                                   interval=10)

    anim.save('test_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
