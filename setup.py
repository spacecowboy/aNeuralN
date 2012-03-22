#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

numpy_include = numpy.get_include()

network_ext = Extension('fast_network',
          sources = ['C_ext/fast_network.c', 'C_ext/activation_functions.c'],
          include_dirs = [numpy_include],
          extra_compile_args = ['-std=c99'])

setup(name = 'aNeuralN',
      version = '0.2',
      description = 'A neural network package.',
      author = 'Jonas Kalderstam',
      author_email = 'jonas@kalderstam.se',
      url = 'https://github.com/spacecowboy/aNeuralN',
      packages = ['kalderstam', 'kalderstam.matlab', 'kalderstam.matlab.tests',
                  'kalderstam.neural', 'kalderstam.neural.tests',
                  'kalderstam.neural.error_functions', 'kalderstam.neural.error_functions.tests',
                  'kalderstam.neural.gui', 'kalderstam.neural.training',
                  'kalderstam.neural.training.tests', 'kalderstam.util',
                  'kalderstam.util.tests'],
      package_dir = {'': '.'},
      package_data = {'kalderstam.neural.gui': ['*.glade']},
      ext_package = 'kalderstam.neural',
      ext_modules = [network_ext],
      requires = ['numpy'],
     )
