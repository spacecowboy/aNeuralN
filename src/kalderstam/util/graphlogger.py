'''
This class graphs data instead of printing it as the logger does.
'''

import numpy as np
import matplotlib.pyplot as plt

loggers = {}
info = 1
debug = 2
logging_level = info

def getGraphLogger(name = None, style = 'gs'):
    if name not in loggers:
        loggers[name] = graphlogger(name, style)
    return loggers[name]

def set_logging_level(level):
    logging_level = level

class graphlogger():
    def __init__(self, name, style):
        self.name = name
        self.x_values = []
        self.y_values = []
        #pyplot stuff
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(str(self.name))
        self.line, = self.ax.plot([], [], style)
        self.ax.grid()
        self.clean_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.set_ylim(-0.0001, 0.0001)
        
    def infoplot(self, *args, **kwargs):
        if logging_level <= info:
            self.plot(*args, **kwargs)
    
    def debugplot(self, *args, **kwargs):
        if logging_level <= debug:
            self.plot(*args, **kwargs)
            
    def show(self):
        self.fig.canvas.draw()
        plt.show()
    
    def interactive(self, on=True):
        if on:
            plt.ion()
        else:
            plt.ioff()
            
    def plot(self, y_val, x_val = None):
        if x_val == None:
            if len(self.x_values) == 0:
                x_val = 0
            else:
                x_val = self.x_values[len(self.x_values) - 1] + 1
                
        x = float(x_val)
        y = float(y_val)
        
        self.x_values.append(x)
        self.y_values.append(y)
        
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        reclear = False
        if x >= xmax:
            self.ax.set_xlim(xmin, 2*x)
            reclear = True
        if x < xmin:
            if x >= 0:
                self.ax.set_xlim(x/2, xmax)
            else:
                self.ax.set_xlim(2*x, xmax)
            reclear = True
            
        if y >= ymax:
            self.ax.set_ylim(ymin, 2*y)
            reclear = True
        if y < ymin:
            if y >= 0:
                self.ax.set_ylim(y/2, ymax)
            else:
                self.ax.set_ylim(2*y, ymax)
            reclear = True
        
#        if reclear:
#            self.fig.canvas.draw()
#            self.clean_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            
        #self.line, = self.ax.plot(self.x_values, self.y_values, 'gs')
        self.line.set_data(self.x_values, self.y_values)
        
#        if self.draw_initial:
#            self.fig.canvas.draw()
#            self.draw_initial = False
#        
#        # just draw the animated artist
#        self.ax.draw_artist(self.line)
#        # just redraw the axes rectangle
#        self.fig.canvas.blit(self.ax.bbox)
        
if __name__ == '__main__':
    logger = getGraphLogger('TestLogging', 'b-')
    set_logging_level(debug)
    #logger.interactive()
    for x in range(100):
        y = 0.2*np.sin(x)
        logger.infoplot(y_val = y)
    logger.show()