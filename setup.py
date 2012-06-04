#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

numpy_include = numpy.get_include()

node_ext = Extension('fast_node',
          sources = ['ann/network/fast_node.c', 'ann/network/activation_functions.c'],
          include_dirs = [numpy_include],
          extra_compile_args = ['-std=c99'])

network_ext = Extension('fast_network',
          sources = ['ann/network/fast_network.c', 'ann/network/fast_node.c'],
          include_dirs = [numpy_include],
          extra_compile_args = ['-std=c99'])

setup(name = 'aNeuralN',
      version = '0.6',
      description = 'A neural network package.',
      author = 'Jonas Kalderstam',
      author_email = 'jonas@kalderstam.se',
      url = 'https://github.com/spacecowboy/aNeuralN',
      packages = ['ann', 'ann.network',
                  'ann.trainingfunctions'],
      package_dir = {'': ''},
      #package_data = {'kalderstam.neural.gui': ['*.glade']},
      ext_package = 'ann.network',
      ext_modules = [node_ext, network_ext],
      requires = ['numpy'],
     )
