'''
This class graphs data instead of printing it as the logger does.
'''

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None #This makes matplotlib optional
except RuntimeError:
    plt = None
import numpy as np

loggers = {}
nothing = 0
error = 1
info = 2
debug = 3
logging_level = nothing

def getGraphLogger(name = None):
    if name not in loggers:
        loggers[name] = graphlogger(name)
    return loggers[name]

def setLoggingLevel(level):
    global logging_level
    logging_level = level

def debugPlot(name, y, x = None, style = 'b', subset = None):
    '''Name of the graph to plot to, the y value, and an optional x value'''
    getGraphLogger(name).debugPlot(y, x, subset, style)

def infoPlot(name, y, x = None, style = 'b', subset = None):
    '''Name of the graph to plot to, the y value, and an optional x value'''
    getGraphLogger(name).infoPlot(y, x, subset, style)

def setup():
    for name in loggers:
        loggers[name].setup()

def show():
    setup()
    plt.show()

class Subset():
    def __init__(self, style):
        self.x_values = []
        self.y_values = []
        self.style = style

class graphlogger():
    def __init__(self, name):
        self.name = name
        self.subsets = {}

        self.ymin = -0.0001
        self.ymax = 0.0001
        self.xmin = 0
        self.xmax = 1

        self.ready = False

    def infoPlot(self, *args, **kwargs):
        global logging_level
        if logging_level >= info:
            self.plot(*args, **kwargs)

    def debugPlot(self, *args, **kwargs):
        global logging_level
        if logging_level >= debug:
            self.plot(*args, **kwargs)
    def setup(self):
        if not self.ready:
            if plt:
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111)
                self.ax.set_title(str(self.name))
                self.ax.grid()
                self.clean_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
                self.ax.set_xlim(self.xmin, self.xmax)
                self.ax.set_ylim(self.ymin, self.ymax)

                for subset in self.subsets.values():
                    subset.line, = self.ax.plot([], [], subset.style)
                    subset.line.set_data(subset.x_values, subset.y_values)

            self.ready = True

    def show(self):
        self.setup()
        if plt:
            plt.show()

    def plot(self, y_val, x_val = None, subset = None, style = 'g-'):
        if plt:
            self.ready = False #Forces a new figure to be drawn next time since we've added data
            if x_val is None:
                if not self.subsets.has_key(subset):
                    self.subsets[subset] = Subset(style)

                if len(self.subsets[subset].x_values) == 0:
                    x_val = 0
                else:
                    x_val = self.subsets[subset].x_values[len(self.subsets[subset].x_values) - 1] + 1

            x = float(x_val)
            y = float(y_val)

            self.subsets[subset].x_values.append(x)
            self.subsets[subset].y_values.append(y)

            # Change limits if needed

            if x >= self.xmax:
                self.xmax = 2 * x
            if x < self.xmin:
                if x >= 0:
                    self.xmin = x / 2
                else:
                    self.xmin = 2 * x

            if y >= self.ymax:
                self.ymax = 2 * y
            if y < self.ymin:
                if y >= 0:
                    self.ymin = y / 2
                else:
                    self.ymin = y * 2


if __name__ == '__main__':
    logger = getGraphLogger('TestLogging')
    setLoggingLevel(debug)
    #logger.interactive()
    for x in xrange(100):
        y = 0.2 * np.sin(x)
        logger.infoPlot(y_val = y)
    logger.show()

    for x in xrange(100):
        y = np.sin(x) * np.exp(-x / 20)
        debugPlot('Another way to graph', y, style = 'r-')
    show()
