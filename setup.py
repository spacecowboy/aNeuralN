#!/usr/bin/env python

from distutils.core import setup, Extension

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
      ext_modules = [Extension('neural.error_functions.cox_error_in_c', ['src/kalderstam/neural/error_functions/cox_error_in_c.c'])],
      requires = [],
     )
