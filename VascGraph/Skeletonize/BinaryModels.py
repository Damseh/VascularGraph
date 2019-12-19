#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 21:10:46 2019

@author: rdamseh
"""
from VascGraph.Tools.CalcTools import *
import skimage as sk
import scipy as sp
import numpy as np

def Tmodel(noisy=False, smooth=False):
    
    tr1=np.zeros((21,21))
    ind=(np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  3,
         3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
         4,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  8,
         9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13,
        13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17,
        17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20]),
    np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20,  0,  1, 19, 20,  0,  1,  2,  3,  4,
         5,  6,  7,  8,  9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  0,  1,
         2,  3,  4,  5,  6,  7,  8,  9, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20,  8,  9, 11, 12,  8,  9, 11, 12,  8,  9, 11, 12,  8,  9, 11, 12,
         8,  9, 11, 12,  8,  9, 11, 12,  8,  9, 11, 12,  8,  9, 11, 12,  8,
         9, 11, 12,  8,  9, 11, 12,  8,  9, 11, 12,  8,  9, 11, 12,  8,  9,
        11, 12,  8,  9, 11, 12,  8,  9, 10, 11, 12,  8,  9, 10, 11, 12]))
    tr1[ind]=1
    
    ind=(np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,
         3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,
         4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  7,  7,  7,  7,
         7,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 11,
        11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14,
        14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17,
        18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20]),
    np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20,  0,  1,  2,  3,  4,  5,  6,  7,  8,
         9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  0,  1,  2,  3,  4,
         5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  0,
         1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20,  8,  9, 10, 11, 12,  8,  9, 10, 11, 12,  8,  9, 10, 11,
        12,  8,  9, 10, 11, 12,  8,  9, 10, 11, 12,  8,  9, 10, 11, 12,  8,
         9, 10, 11, 12,  8,  9, 10, 11, 12,  8,  9, 10, 11, 12,  8,  9, 10,
        11, 12,  8,  9, 10, 11, 12,  8,  9, 10, 11, 12,  8,  9, 10, 11, 12,
         8,  9, 10, 11, 12,  8,  9, 10, 11, 12,  8,  9, 10, 11, 12]))
    tr2=np.zeros((21,21))
    tr2[ind]=1
    seg=np.zeros((5,21,21))
    
    if noisy:
        seg[(0,1),:,:]= tr1[None,:]
        seg[2]= tr2[None,:]
        seg[(3,4),:,:]= tr1[None,:]
    else:
        seg[:]= tr2[None,:]
   
    seg=sp.ndimage.zoom(seg.astype(int),(3.0,3.0,3.0))
    seg = np.pad(seg, ((1,1),(1,1),(1,1)), mode='constant', constant_values=0)
    
    if smooth:
        seg=sp.ndimage.morphology.binary_erosion(seg.astype(bool), iterations=5)
    
    return seg.astype(int)  





def BarModel(noisy=False):
    seg=np.ones((20,20,100))
    if noisy:
        seg[0:6,4:16,:] = 0
        seg[14:,4:16,:] = 0
    else:
        pass
    return np.pad(seg, ((1,1),(1,1),(1,1)), mode='constant', constant_values=0)



def RectangularModel(noisy=False):
    
    import os
    import sys
    
    cwd=os.path.dirname(os.path.realpath(__file__))
    print(cwd)
    path=cwd+'/models/'
    im=sk.io.imread(path+'rectangle.png')[:,:,3]>0
    im=im.astype(int)
    im=sp.ndimage.zoom(im, (.1,.1))
    im=np.repeat(im[None, :, :], 150, axis=0)
    im=np.pad(im, ((1,1),(1,1),(1,1)), mode='constant', constant_values=0)
    sys.path.pop()
    
    return im


def Rect1Model(noisy=False):
    
    import os
    import sys
    
    cwd=os.path.dirname(os.path.realpath(__file__))
    print(cwd)
    path=cwd+'/models/'
    im=sk.io.imread(path+'rect1.png')[:,:,3]>0
    im=im.astype(int)
    im=sp.ndimage.zoom(im, (.1,.1))
    im=np.repeat(im[None, :, :], 10, axis=0)
    im=np.pad(im, ((1,1),(1,1),(1,1)), mode='constant', constant_values=0)
    sys.path.pop()
    
    return im

def Shape1Model(noisy=False):
    
    import os
    import sys
    
    cwd=os.path.dirname(os.path.realpath(__file__))
    print(cwd)
    path=cwd+'/models/'
    im=sk.io.imread(path+'shape1.png')[:,:,3]>0
    im=im.astype(int)
    im=sp.ndimage.zoom(im, (.05,.05))
    im=np.repeat(im[None, :, :], 200, axis=0)
    im=np.pad(im, ((1,1),(1,1),(1,1)), mode='constant', constant_values=0)
    sys.path.pop()
    
    return im



def GlassesModel(noisy=False):
    
    import os
    import sys
    
    cwd=os.path.dirname(os.path.realpath(__file__))
    print(cwd)
    path=cwd+'/models/'
    im=sk.io.imread(path+'glasses.png')[:,:,3]>0
    im=im.astype(int)
    im=sp.ndimage.zoom(im, (.10,.10))
    im=np.repeat(im[None, :, :], 10, axis=0)
    im=np.pad(im, ((1,1),(1,1),(1,1)), mode='constant', constant_values=0)
    sys.path.pop()
    
    return im


def TwoCirclesModel(noisy=False):
    
    import os
    import sys
    
    cwd=os.path.dirname(os.path.realpath(__file__))
    print(cwd)
    path=cwd+'/models/'
    im=sk.io.imread(path+'twocircles.png')[:,:,3]>0
    im=im.astype(int)
    im=sp.ndimage.zoom(im, (.01,.01))
    im=np.repeat(im[None, :, :], 150, axis=0)
    im=np.pad(im, ((1,1),(1,1),(1,1)), mode='constant', constant_values=0)
    sys.path.pop()
    
    return im


def Circles2Model(noisy=False):
    
    import os
    import sys
    
    cwd=os.path.dirname(os.path.realpath(__file__))
    print(cwd)
    path=cwd+'/models/'
    im=sk.io.imread(path+'circle2.png')[:,:,3]>0
    im=im.astype(int)
    im=sp.ndimage.zoom(im, (.01,.01))
    im=np.repeat(im[None, :, :], 150, axis=0)
    im=np.pad(im, ((1,1),(1,1),(1,1)), mode='constant', constant_values=0)
    sys.path.pop()
    
    return im


def XModel(noisy=False):
    
    import os
    import sys
    
    cwd=os.path.dirname(os.path.realpath(__file__))
    print(cwd)
    path=cwd+'/models/'
    im=sk.io.imread(path+'x.png')[:,:,3]>0
    im=im.astype(int)
    im=sp.ndimage.zoom(im, (.075,.075))
    im=np.repeat(im[None, :, :], 150, axis=0)
    im=np.pad(im, ((1,1),(1,1),(1,1)), mode='constant', constant_values=0)
    sys.path.pop()
    
    return im
       
    
def CircleModel(noisy=False):
    
    import os
    import sys
    
    cwd=os.path.dirname(os.path.realpath(__file__))
    print(cwd)
    path=cwd+'/models/'
    im=sk.io.imread(path+'circle.png')[:,:,3]>0
	
    im=sp.ndimage.zoom(im, (.075,.075))
    im=np.repeat(im[None, :, :], 150, axis=0)
    im=np.pad(im, ((1,1),(1,1),(1,1)), mode='constant', constant_values=0)
    sys.path.pop()
    
    return im
       
    
def SquareModel(noisy=False):
    
    import os
    import sys
    
    cwd=os.path.dirname(os.path.realpath(__file__))
    print(cwd)
    path=cwd+'/models/'
    im=sk.io.imread(path+'square.png')[:,:,3]>0
	
    im=sp.ndimage.zoom(im, (.075,.075))
    im=np.repeat(im[None, :, :], 150, axis=0)
    im=np.pad(im, ((1,1),(1,1),(1,1)), mode='constant', constant_values=0)
    sys.path.pop()
    
    return im    
    
    
    
    
    
    
    
