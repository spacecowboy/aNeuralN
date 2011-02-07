import sys
from kalderstam.neural.network import build_feedforward
from kalderstam.neural.functions.activation_functions import linear, logsig, tanh
from kalderstam.neural.matlab_functions import loadsyn1, plotroc, plot2d2c, stat
from kalderstam.util.filehandling import parse_file
from kalderstam.neural.functions import training_functions
from kalderstam.neural.trainer import Trainer
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
    
class NeuralUI():
    def __init__(self):
        #Set the Glade file
        self.gladefile = "NeuralGUI.glade"
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
        
        
    def on_startbutton_pressed(self, *args):
        logging.basicConfig(level=logging.DEBUG)
        
        P, T = loadsyn1(100)
        #P, T = parse_file(self.input_entry.get_text(), self.input_number.get_value(), self.output_number.get_value())
        
        trainer = Trainer()
        trainer.input_number = self.input_number.get_value()
        trainer.hidden_number = self.hidden_number.get_value()
        trainer.output_number = self.output_number.get_value()
        trainer.hidden_activation_function = self.hidden_activation_function
        trainer.output_activation_function = self.output_activation_function
        trainer.training_method = self.training_method
        trainer.epochs = self.epoch_number.get_value()
        trainer.inputs = P
        trainer.outputs = T
        
        trainer.start()
        
            
    def on_MainWindow_destroy(self, *args):
        gtk.main_quit()
    
    def on_check_train_toggled(self, chk_button):
        if chk_button.props.active:
            self.trainingbox.props.sensitive = True
        else:
            self.trainingbox.props.sensitive = False
            
    def set_training_gradient(self, radio_button):
        if radio_button.props.active:
            self.training_method = training_functions.traingd
    
    def set_training_genetic(self, radio_button):
        if radio_button.props.active:
            self.training_method = training_functions.train_evolutionary
    
    def set_weight_update_online(self, radio_button):
        if radio_button.props.active:
            self.weight_update_method = 'online'
    
    def set_error_sumsquare(self, radio_button):
        if radio_button.props.active:
            pass
            
    def set_hidden_activation_linear(self, radio_button):
        if radio_button.props.active: 
            self.hidden_activation_function = linear()
            
    def set_hidden_activation_logsig(self, radio_button):
        if radio_button.props.active: 
            self.hidden_activation_function = logsig()
            
    def set_hidden_activation_tanh(self, radio_button):
        if radio_button.props.active: 
            self.hidden_activation_function = tanh()
            
    def set_output_activation_linear(self, radio_button):
        if radio_button.props.active: 
            self.output_activation_function = linear()
            
    def set_output_activation_logsig(self, radio_button):
        if radio_button.props.active: 
            self.output_activation_function = logsig()
            
    def set_output_activation_tanh(self, radio_button):
        if radio_button.props.active: 
            self.output_activation_function = tanh()
            
    def on_file_chosen_button(self, button):
        self.input_entry.set_text(button.get_filename())
        
    def on_config_file_button(self, button):
        print button.get_filename()
        self.config_entry.set_text(button.get_filename())
        
def main():
    logging.basicConfig(level=logging.DEBUG)
    gui = NeuralUI()
    gui.window.show()
    gtk.main()
            
if __name__ == '__main__':
    main()