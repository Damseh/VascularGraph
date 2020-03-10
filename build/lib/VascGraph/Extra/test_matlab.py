#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:53:25 2019

@author: rdamseh
"""

import matlab.engine as eng
from pymatbridge import Matlab

mlab = Matlab()
mlab.start()
mlab.run_code('x=[1,2,3,4,5,6].^2')
x=mlab.get_variable('x')