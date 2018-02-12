class SubplotAnimation(animation.TimedAnimation):
    def __init__(self):
        self.fig = plt.figure(figsize=(6,6))
        self.fig.suptitle('Multivariate Analysis', x=0.38, y=0.85, 
                         horizontalalignment='center',
                         fontsize=16,
                         fontweight='bold')
        # Subplot layout       
        self.gs = gridspec.GridSpec(6,6)
        self.gs.update(left=0.1,  
                       right=0.95, 
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

        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
        
        scale = p.StandardScaler().fit(x_train)
        self.data_series = self.create_transform(scale.transform(x_train), 
                                                 y_train, 
                                                 time_steps=60, 
                                                 delay=40)        
            
        
        self.t = np.linspace(0, 80, 400)
        self.x = np.cos(2 * np.pi * self.t / 10.)
        self.y = np.sin(2 * np.pi * self.t / 10.)
        self.z = 10 * self.t
#
#        self.ax1.set_xlabel('x')
#        self.ax1.set_ylabel('y')
        self.line1 =  Line2D([], [], color='black')
        self.line1a = Line2D([], [], color='red', linewidth=2)
        self.line1e = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        self.ax1.add_line(self.line1)
        self.ax1.add_line(self.line1a)
        self.ax1.add_line(self.line1e)
#        self.ax1.set_xlim(-1, 1)
#        self.ax1.set_ylim(-2, 2)
#        self.ax1.set_aspect('equal', 'datalim')
#
#        self.ax2.set_xlabel('y')
#        self.ax2.set_ylabel('z')
        self.line2 = Line2D([], [], color='black')
        self.line2a = Line2D([], [], color='red', linewidth=2)
        self.line2e = Line2D(
            [], [], color='red', marker='o', markeredgecolor='r')
        self.ax2.add_line(self.line2)
        self.ax2.add_line(self.line2a)
        self.ax2.add_line(self.line2e)
        
#        self.ax2.set_xlim(-1, 1)
#        self.ax2.set_ylim(0, 800)
#        self.ax3.set_xlabel('x')
#        self.ax3.set_ylabel('z')
        
        self.line3 =  Line2D([], [], color='black')
        self.line3a = Line2D([], [], color='red', linewidth=2)
        self.line3e = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        self.ax3.add_line(self.line3)
        self.ax3.add_line(self.line3a)
        self.ax3.add_line(self.line3e)
#        self.ax3.set_xlim(-1, 1)
#        self.ax3.set_ylim(0, 800)

        self._create_histograms()
        self._create_scatter()
        animation.TimedAnimation.__init__(self, self.fig, interval=50, blit=True)


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
        x_top = xn  # freq[0]
        y_top = yn  # freq[1]
     
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


    def _create_scatter(self):
        
        self.x1_ttl = self.ax1.text(.5, -0.1, '', 
                              horizontalalignment='center',
                              transform = self.ax1.transAxes, 
                              fontweight='bold', fontsize=12)
        
        self.y1_ttl = self.ax1.text(-0.1, .5, '', 
                              transform = self.ax1.transAxes, 
                              horizontalalignment='left',
                              verticalalignment='center',
                              rotation=90, 
                              fontweight='bold', # bbox=dict(facecolor='red', alpha=0.5),
                              fontsize=12)
        self.line = [self.ax1.plot(
                         self.data_series[0][2], 
                         self.data_series[0][3], 
                         'o', c='#FA6367',
                         markerfacecolor='#FEEAA8', 
                         markersize=5)]
            
            
    def _draw_frame(self, framedata):
        i = framedata
        head = i - 1
        head_slice = (self.t > self.t[i] - 1.0) & (self.t < self.t[i])

        self.line1.set_data(self.x[:i], self.y[:i])
        self.line1a.set_data(self.x[head_slice], self.y[head_slice])
        self.line1e.set_data(self.x[head], self.y[head])

        self.line2.set_data(self.y[:i], self.z[:i])
        self.line2a.set_data(self.y[head_slice], self.z[head_slice])
        self.line2e.set_data(self.y[head], self.z[head])

        self.line3.set_data(self.x[:i], self.z[:i])
        self.line3a.set_data(self.x[head_slice], self.z[head_slice])
        self.line3e.set_data(self.x[head], self.z[head])

        
        self._drawn_artists = [self.line1, self.line1a, self.line1e,
                               self.line2, self.line2a, self.line2e,
                               self.line3, self.line3a, self.line3e]

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines = [self.line1, self.line1a, self.line1e,
                 self.line2, self.line2a, self.line2e,
                 self.line3, self.line3a, self.line3e]
        for l in lines:
            l.set_data([], [])

    def create_transform(self, data, labels, time_steps=20, delay=200):
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
                
    
ani = SubplotAnimation()
# ani.save('test_sub.mp4')
plt.show()