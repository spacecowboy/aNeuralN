#Expose the appropriate version to the rest of the module
try:
    from fast_network import Node, BiasNode
except ImportError as e:
    print(e)
    print("Not installed, using python version...")
    from pynodemodule import Node, BiasNode
