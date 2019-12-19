#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:51:49 2019

@author: rdamseh
"""

im2d=sp.misc.imread('/home/rdamseh/GraphPaper2018V1'+'/flow/test_network.png')[:,:,3]>0
print(np.sum(im2d[:]))
im3d=np.dstack([im2d]*10).astype('int')
im3d=np.rollaxis(im3d,2,0)
im3d=np.pad(im3d, ((1,1),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))