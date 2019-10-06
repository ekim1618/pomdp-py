#!/usr/bin/env python

from distutils.core import setup

setup(name='pomdp-py',
      packages=['pomdp_py'],
      version='1.0',
      description='POMDP, OO-POMDP for python with a parser.',
      install_requires=[
          'simple_rl',     # make sure to use zkytony/simple_rl
          'numpy',
          'matplotlib',
          'pygame',        # for some tests
          'opencv-python'  # for some tests
      ],
      author='Kaiyu Zheng',
      author_email='kaiyutony@gmail.com',
      keywords = ['Partially Observable Markov Decision Process', 'POMDP'],
     )
