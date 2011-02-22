import sys
from kalderstam.neural.network import build_feedforward
from kalderstam.neural.activation_functions import linear, logsig, tanh
try:
    import pygtk
    pygtk.require("2.0")
    import logging
except:
    pass
try:
    import gtk
    import gtk.glade
except:
    sys.exit(1)
    
class ANN_Creator():
    def __init__(self, path):
        self.net = None
        #Set the Glade file
        self.gladefile = path + "New_ANN.glade"
        self.builder = gtk.Builder()
        self.builder.add_from_file(self.gladefile)
        
        #Connect methods with signals
        self.builder.connect_signals(self)
        #Get the Main Window, and connect the "destroy" event
        self.window = self.builder.get_object("window1")
        
        self.input_number = self.builder.get_object("input_adjuster")
        self.hidden_number = self.builder.get_object("hidden_adjuster")
        self.output_number = self.builder.get_object("output_adjuster")
        
        #default values
        self.hidden_activation_function = tanh()
        self.output_activation_function = logsig()
        
    def on_create_button_clicked(self, *args):
        logging.basicConfig(level=logging.DEBUG)
        
        input_number = int(self.input_number.get_value())
        hidden_number = int(self.hidden_number.get_value())
        output_number = int(self.output_number.get_value())
        
        self.net = build_feedforward(input_number, hidden_number, output_number, self.hidden_activation_function, self.output_activation_function)
        
        #Show ANN-window
        #handled when net is returned
        
        #Kill this window
        self.window.hide()
        gtk.main_quit()
            
    def on_window1_destroy(self, *args):
        gtk.main_quit()
            
    def on_activation_hidden_linear_toggled(self, radio_button):
        if radio_button.props.active: 
            self.hidden_activation_function = linear()
            
    def on_activation_hidden_logsig_toggled(self, radio_button):
        if radio_button.props.active: 
            self.hidden_activation_function = logsig()
            
    def on_activation_hidden_tanh_toggled(self, radio_button):
        if radio_button.props.active: 
            self.hidden_activation_function = tanh()
            
    def on_activation_output_linear_toggled(self, radio_button):
        if radio_button.props.active: 
            self.output_activation_function = linear()
            
    def on_activation_output_logsig_toggled(self, radio_button):
        if radio_button.props.active: 
            self.output_activation_function = logsig()
            
    def on_activation_output_tanh_toggled(self, radio_button):
        if radio_button.props.active: 
            self.output_activation_function = tanh()
        
def show_new_ann_window(path):
    logging.basicConfig(level=logging.DEBUG)
    gui = ANN_Creator(path)
    gui.window.show()
    gtk.main()
    return gui.net
            
if __name__ == '__main__':
    show_new_ann_window(sys.argv[0][0:-len("New_ANN.py")])