'''
This class graphs data instead of printing it as the logger does.
'''

import matplotlib.pyplot as plt
import numpy as np

loggers = {}
nothing = 0
error = 1
info = 2
debug = 3
logging_level = nothing

def getGraphLogger(name = None, style = 'gs'):
    if name not in loggers:
        loggers[name] = graphlogger(name, style)
    return loggers[name]

def setLoggingLevel(level):
    global logging_level
    logging_level = level

def debugPlot(name, y, x = None, style = 'b'):
    '''Name of the graph to plot to, the y value, and an optional x value'''
    getGraphLogger(name, style).debugPlot(y, x)

def infoPlot(name, y, x = None, style = 'b'):
    '''Name of the graph to plot to, the y value, and an optional x value'''
    getGraphLogger(name, style).infoPlot(y, x)

def show():
    plt.show()

class graphlogger():
    def __init__(self, name, style):
        self.name = name
        self.x_values = []
        self.y_values = []

        self.style = style
        self.ready = False

    def setup(self):
        if not self.ready:
            self.ready = True

            #pyplot stuff
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title(str(self.name))
            self.line, = self.ax.plot([], [], self.style)
            self.ax.grid()
            self.clean_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.ax.set_ylim(-0.0001, 0.0001)

    def infoPlot(self, *args, **kwargs):
        global logging_level
        if logging_level >= info:
            self.plot(*args, **kwargs)

    def debugPlot(self, *args, **kwargs):
        global logging_level
        if logging_level >= debug:
            self.plot(*args, **kwargs)

    def show(self):
        #self.fig.canvas.draw()
        plt.show()

    def plot(self, y_val, x_val = None):
        self.setup()

        if x_val is None:
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
            self.ax.set_xlim(xmin, 2 * x)
            reclear = True
        if x < xmin:
            if x >= 0:
                self.ax.set_xlim(x / 2, xmax)
            else:
                self.ax.set_xlim(2 * x, xmax)
            reclear = True

        if y >= ymax:
            self.ax.set_ylim(ymin, 2 * y)
            reclear = True
        if y < ymin:
            if y >= 0:
                self.ax.set_ylim(y / 2, ymax)
            else:
                self.ax.set_ylim(2 * y, ymax)
            reclear = True

        self.line.set_data(self.x_values, self.y_values)

if __name__ == '__main__':
    logger = getGraphLogger('TestLogging', 'b-')
    setLoggingLevel(debug)
    #logger.interactive()
    for x in range(100):
        y = 0.2 * np.sin(x)
        logger.infoPlot(y_val = y)
    logger.show()

    for x in range(100):
        y = np.sin(x) * np.exp(-x / 20)
        debugPlot('Another way to graph', y, style = 'r-')
    show()
