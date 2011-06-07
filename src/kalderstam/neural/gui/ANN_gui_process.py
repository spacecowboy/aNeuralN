import sys
try:
    import pygtk
    pygtk.require("2.0")
    import logging
    from multiprocessing import Process
    from threading import Thread
    import gtk
    import gtk.glade
    from kalderstam.neural import training_functions
except:
    sys.exit(1)

#This gui is a process, because it should not even run on the same cpu as other networks preferably. The network is run through a different thread inside the process
class ANN_gui(Process):
    def __init__(self):
        Process.__init__(self)

        #Set the Glade file
        self.gladefile = "ANN.glade"
        self.builder = gtk.Builder()
        self.builder.add_from_file(self.gladefile)

        #Connect methods with signals
        self.builder.connect_signals(self)
        #Get the Main Window, and connect the "destroy" event
        self.window = self.builder.get_object("MainWindow")
        self.trainingbox = self.builder.get_object("trainingbox")
        self.input_number = self.builder.get_object("input_number")
        self.hidden_number = self.builder.get_object("hidden_number")
        self.output_number = self.builder.get_object("output_number")
        self.epoch_number = self.builder.get_object("epoch_number")

        self.input_entry = self.builder.get_object("input_entry")
        self.config_entry = self.builder.get_object("config_entry")

        #default values
        self.hidden_activation_function = tanh()
        self.output_activation_function = logsig()
        self.training_method = training_functions.traingd

    #Shows and starts the window
    def run(self):
        self.window.show()
        gtk.main()

if __name__ == '__main__':
    pass
