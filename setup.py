#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

numpy_include = numpy.get_include()

network_ext = Extension('fast_network',
          sources = ['C_ext/fast_network.c', 'C_ext/activation_functions.c'],
          include_dirs = [numpy_include],
          extra_compile_args = ['-std=c99'])

setup(name = 'aNeuralN',
      version = '0.4',
      description = 'A neural network package.',
      author = 'Jonas Kalderstam',
      author_email = 'jonas@kalderstam.se',
      url = 'https://github.com/spacecowboy/aNeuralN',
      packages = ['ann',
                  'ann.trainingfunctions'],
      package_dir = {'': ''},
      #package_data = {'kalderstam.neural.gui': ['*.glade']},
      ext_package = 'ann',
      ext_modules = [network_ext],
      requires = ['numpy'],
     )
