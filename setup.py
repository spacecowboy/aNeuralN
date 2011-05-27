#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

numpy_include = numpy.get_include()

cox_ext = Extension('neural.error_functions.cox_error_in_c',
          sources = ['src/kalderstam/neural/error_functions/cox_error_in_c.c'],
          include_dirs = [numpy_include],
          extra_compile_args = ['-std=c99'])

setup(name = 'aNeuralN',
      version = '1.0',
      description = 'A neural network package.',
      author = 'Jonas Kalderstam',
      author_email = 'jonas@kalderstam.se',
      url = 'https://github.com/spacecowboy/aNeuralN',
      packages = ['kalderstam'],
      package_dir = {'kalderstam': 'src/kalderstam'},
      package_data = {'kalderstam.neural.gui': ['src/kalderstam/neural/gui/*.glade']},
      ext_package = 'kalderstam',
      ext_modules = [cox_ext],
      requires = ['numpy', 'matplotlib'],
     )
