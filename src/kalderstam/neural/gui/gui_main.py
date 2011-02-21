from kalderstam.util.filehandling import load_network
from kalderstam.neural.gui.ANN import show_ann_window
from kalderstam.neural.gui.New_ANN import show_new_ann_window
import sys

def main(path = None):
    if path:
        show_ann_window(load_network(path))
    else:
        net = show_new_ann_window()
        if net:
            show_ann_window(net)
            
if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()