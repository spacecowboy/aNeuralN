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

def main():
    logging.basicConfig(level=logging.DEBUG)
    gui = NeuralUI()
    gui.window.show()
    gtk.main()
            
if __name__ == '__main__':
    main()