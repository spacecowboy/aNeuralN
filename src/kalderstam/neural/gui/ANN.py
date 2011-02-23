import sys
from kalderstam.neural.network import build_feedforward
from kalderstam.neural.activation_functions import linear, logsig, tanh
from kalderstam.neural.matlab_functions import loadsyn1, plotroc, plot2d2c, stat
from kalderstam.util.filehandling import parse_file
from kalderstam.neural import training_functions
from kalderstam.neural.gui.New_ANN import show_new_ann_window
from kalderstam.neural.gui.Dialogs import show_open_dialog, show_save_dialog
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
    
class ANN_gui():
    def __init__(self, path, net):
        self.net = net
        #Set the Glade file
        self.path = path
        self.gladefile = path + "ANN.glade"
        self.builder = gtk.Builder()
        self.builder.add_from_file(self.gladefile)
        
        #Connect methods with signals
        self.builder.connect_signals(self)
        #Get the Main Window, and connect the "destroy" event
        self.window = self.builder.get_object("window1")
        self.trainingbox = self.builder.get_object("trainingbox")
#        self.input_number = self.builder.get_object("input_number")
#        self.hidden_number = self.builder.get_object("hidden_number")
        self.trn_btn_gradient = self.builder.get_object("trn_btn_gradient")
        self.trn_btn_genetic = self.builder.get_object("trn_btn_genetic")
#        self.output_number = self.builder.get_object("output_number")
        self.epoch_number = self.builder.get_object("epoch_adjuster")
        self.block_size = self.builder.get_object("block_size_adjuster")
        
        #Gradient Descent
        self.learning_rate = self.builder.get_object("learning_rate_adjuster")
        self.momentum = self.builder.get_object("momentum_adjuster")
        #Genetic
        self.population = self.builder.get_object("population_adjuster")
        self.random_range = self.builder.get_object("random_range_adjuster")
        self.mutation = self.builder.get_object("mutation_adjuster")
        
        self.train_button = self.builder.get_object("train_button")
        self.stop_button = self.builder.get_object("stop_button")
        self.sim_button = self.builder.get_object("sim_button")
        
        self.inputs_entry = self.builder.get_object("inputs_entry")
        self.ignore_entry = self.builder.get_object("ignore_entry")
        self.targets_entry = self.builder.get_object("targets_entry")
        
        self.network_image = self.builder.get_object("network_image")
#        self.config_entry = self.builder.get_object("config_entry")
        
        #default values
        self.trainer = None
        
    def visualize_network(self):
        #self.__get_trained_network()
        #pixmap = gtk.gdk.pixmap_create_from_data(None, data, width, height, depth = 8, fg, bg)
        #self.network_image.set_from_pixmap(pixmap)
        pass
        
    def on_train_button_pressed(self, *args):
        #Make the button unpressable
        self.train_button.props.sensitive = False
        self.stop_button.props.sensitive = True
        self.sim_button.props.sensitive = False
        
        P, T = self.read_input_file()
        
        #Set the function and start training with appropriate arguments
        if self.trn_btn_gradient.props.active:
            self.net = training_functions.traingd_block(net = self.net, input_array = P, output_array = T, epochs=self.epoch_number.get_value(), learning_rate=self.learning_rate.get_value(), block_size=self.block_size.get_value(), momentum=self.momentum.get_value())
        elif self.trn_btn_genetic.props.active:
            self.net = training_functions.train_evolutionary(net = self.net, input_array = P, output_array = T, epochs=self.epoch_number.get_value(), population_size = self.population.get_value(), mutation_chance = self.mutation.get_value(), random_range=self.random_range.get_value())
            
        #For single threaded
        self.train_button.props.sensitive = True
        self.stop_button.props.sensitive = False
        self.sim_button.props.sensitive = True

    
    def on_stop_button_pressed(self, *args):
        self.train_button.props.sensitive = True
        self.stop_button.props.sensitive = False
        self.sim_button.props.sensitive = True
    
    def get_cols(self, text):
        targets = []
        for target in text.split():
            targets.append(int(target))
        return targets
    
    def get_target_cols(self):
        return self.get_cols(self.targets_entry.get_text())
    
    def get_ignore_cols(self):
        return self.get_cols(self.ignore_entry.get_text())
    
    def get_input_cols(self):
        return self.get_cols(self.inputs_entry.get_text())
    
    def read_input_file(self):
        targets = self.get_target_cols()
        inputs = self.get_input_cols()
        ignores = self.get_ignore_cols()

        P, T = parse_file(self.input_file, targetcols = targets, inputcols = inputs, ignorecols = ignores)
        
        return (P, T)
    
    def on_sim_button_pressed(self, *args):
        P, T = self.read_input_file()
        
        results = self.net.sim(P)
        
        print("{0:<9}   {1:<9}".format("T", "Output"))
        for index in range(len(P)):
            print("{0:<9}   {1:<9}".format(T[index], results[index]))
        print("\n")
        
        plotroc(results, T)
        plt.show()
            
    def on_window1_destroy(self, *args):
        gtk.main_quit()
    
    def on_check_train_toggled(self, chk_button):
        if chk_button.props.active:
            self.trainingbox.props.sensitive = True
        else:
            self.trainingbox.props.sensitive = False
            
    def set_training_gradient(self, radio_button):
        if radio_button.props.active:
            self.training_method = training_functions.traingd_block
    
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
        self.input_file = button.get_filename()
        
    def on_config_file_button(self, button):
        print button.get_filename()
        self.config_entry.set_text(button.get_filename())
        
    def on_new_item_activate(self, *args):
        net = show_new_ann_window(self.path)
        show_ann_window(self.path, net)
        
    def on_open_item_activate(self, *args):
        net = show_open_dialog(self.path)
        show_ann_window(self.path, net)
        
    def on_save_item_activate(self, *args):
        self.on_save_as_item_activate()
        
    def on_save_as_item_activate(self, *args):
        show_save_dialog(self.path, self.net)
    
    def on_quit_item_activate(self, *args):
        self.window.hide()
        gtk.main_quit()
        
        
def show_ann_window(path, net):
    logging.basicConfig(level=logging.DEBUG)
    gui = ANN_gui(path, net)
    gui.window.show()
    gtk.main()
            
if __name__ == '__main__':
    show_ann_window(sys.argv[0][0:-len("ANN.py")], build_feedforward())