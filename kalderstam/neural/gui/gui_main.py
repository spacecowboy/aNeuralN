from kalderstam.util.filehandling import load_network
from kalderstam.neural.gui.ANN import show_ann_window
from kalderstam.neural.gui.New_ANN import show_new_ann_window
import sys

def main(path, networkpath = None):
    if networkpath:
        show_ann_window(path, load_network(networkpath))
    else:
        net = show_new_ann_window(path)
        if net:
            show_ann_window(path, net)
            
if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(sys.argv)
        main(sys.argv[0][0:-len("gui_main.py")], sys.argv[1])
    else:
        print(sys.argv)
        main(sys.argv[0][0:-len("gui_main.py")])