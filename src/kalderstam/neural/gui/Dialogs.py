import sys
from kalderstam.neural.network import build_feedforward
from kalderstam.neural.activation_functions import linear, logsig, tanh
from kalderstam.neural.matlab_functions import loadsyn1, plotroc, plot2d2c, stat
from kalderstam.util.filehandling import parse_file, load_network, save_network
from kalderstam.neural import training_functions
from kalderstam.neural.gui.New_ANN import show_new_ann_window
try:
    import pygtk
    pygtk.require("2.0")
    import logging
    import matplotlib.pyplot as plt
except:
    pass
try:
    import gtk
    import gtk.glade
except:
    sys.exit(1)
    
class open_dialog():
    def __init__(self):
        self.net = None
        #Set the Glade file
        self.gladefile = "OpenDialog.glade"
        self.builder = gtk.Builder()
        self.builder.add_from_file(self.gladefile)
        
        #Connect methods with signals
        self.builder.connect_signals(self)
        #Get the Main Window, and connect the "destroy" event
        self.window = self.builder.get_object("filechooserdialog1")
        
    def on_cancelButton_clicked(self, *args):
        self.on_filechooserdialog1_destroy()
    
    def on_openButton_clicked(self, *args):
        self.net = load_network(self.window.get_filename())
        #Kill this window
        self.window.hide()
        gtk.main_quit()
            
    def on_filechooserdialog1_destroy(self, *args):
        gtk.main_quit()
        
class save_dialog():
    def __init__(self, net):
        self.net = net
        #Set the Glade file
        self.gladefile = "SaveDialog.glade"
        self.builder = gtk.Builder()
        self.builder.add_from_file(self.gladefile)
        
        #Connect methods with signals
        self.builder.connect_signals(self)
        #Get the Main Window, and connect the "destroy" event
        self.window = self.builder.get_object("filechooserdialog1")
        
    def on_cancelButton_clicked(self, *args):
        self.on_filechooserdialog1_destroy()
    
    def on_saveButton_clicked(self, *args):
        save_network(self.net, filename = self.window.get_filename())
        #Kill this window
        self.window.hide()
        gtk.main_quit()
            
    def on_filechooserdialog1_destroy(self, *args):
        gtk.main_quit()
        
def show_open_dialog():
    logging.basicConfig(level=logging.DEBUG)
    gui = open_dialog()
    gui.window.show()
    gtk.main()
    return gui.net

def show_save_dialog(net):
    logging.basicConfig(level=logging.DEBUG)
    gui = save_dialog(net)
    gui.window.show()
    gtk.main()
            
if __name__ == '__main__':
    net = show_open_dialog()
    show_save_dialog(net)