#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:03:18 2019

@author: rdamseh
"""

from VascGraph.Tools.CalcTools import *
import scipy.io as sio


class ReadStackMat():
    
    def __init__(self, filename=None):
        
        self.filename=filename
        self.__Stack=None
        
 
    def __ReadFile(self):
        mat=sio.loadmat(self.filename)    
        for i in mat.keys():
            if type(mat[i]) is np.ndarray:
                return mat[i]
         
    def GetOutput(self):
        
        try:
            return self.__ReadFile()
        except: 
            print('Cannot read mat file!') 
        
        