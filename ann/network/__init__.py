#Expose the appropriate version to the rest of the module
try:
    from fast_node import Node, BiasNode
except ImportError as e:
    print(e)
    print("Not installed, using python version...")
    from pynode import Node, BiasNode
