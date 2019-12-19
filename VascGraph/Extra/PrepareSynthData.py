#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 12:15:47 2018

@author: rdamseh
"""

from util_validation import *
import scipy.io as cio
import skimage.io as sio
from skimage import morphology as morph
import SimpleITK as sitk
import skimage.morphology as skm
from scipy.ndimage import filters as f
from scipy.ndimage.interpolation import zoom

def binRead(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return ct_scan.astype('bool')


def binDilate(x, p):
    for i in range(p):
        x=skm.binary_dilation(x)      
    return x
 
    
def binBoundary(x):
    
    '''
    Return the boundary of abinary image
    
    Input:
        x: binary image of type 'bool'
    '''
    
    
    xf=x*900+100
    xf=f.uniform_filter(xf)    
    background=(xf<300)
    foreground=(xf>600)
    boundary=np.bitwise_not(np.bitwise_or(background, foreground))
    
    return boundary

def binNoise2(x, level):
    
    s=np.shape(x)
    indx=np.where(x>0)
    indx=[(indx[0][i], indx[1][i], indx[2][i]) for i in range(len(indx[0]))]

    noise=np.random.rand(s[0], s[1], s[2])>.95
    indn=np.where(noise>0)
    indn=[(indn[0][i], indn[1][i], indn[2][i]) for i in range(len(indn[0]))]

    ind=set(indx).intersection(set(indn))
    ind=np.array(list(ind))
    
    
    for i in range(level):
        v=np.zeros(len(ind)).astype('int8')
        l=i+1
        ind1=ind+np.array([v+l, v, v]).T
        ind2=ind+np.array([v-l, v, v]).T
        ind3=ind+np.array([v, v+l, v]).T
        ind4=ind+np.array([v, v-l, v]).T
        ind5=ind+np.array([v, v, v+l]).T
        ind6=ind+np.array([v, v, v-l]).T 
        ind=np.vstack((ind, ind1, ind2, ind3, ind4, ind5, ind6))
    
    ind=[i for i in ind if i[0]<s[0] and i[1]<s[1] and i[2]<s[2]]
    ind=np.array(ind)
    ind=(ind[:,0],ind[:,1],ind[:,2])
    
    x[ind]=1
    
    return x


def binNoise(x, level):
    
    # get all indices
    b=binBoundary(im)
    #visStack(255*b.astype('float'))
    indx=np.where(b>0)
    indx=np.array(indx).T

    # pick random indices from boundary
    noise=np.random.rand(len(indx))>level
    indn=indx[np.where(noise>0)]
    
    
    # get binary image of sampled boundary
    indn=(indn[:,0],indn[:,1],indn[:,2])
    x=b
    x[indn]=0
    #visStack(255*x.astype('float'))

    #apply noise
    sphere=skm.ball(radius=2)
    xx=sc.ndimage.filters.convolve(x.astype('float'), sphere.astype('float'))
    xx=xx>0
    #visStack(255*xx.astype('float'))
    #final image
    imnoisy=(xx+im)>0
    imnoisy=morph.remove_small_holes(imnoisy, 125)

    #visStack(255*imnoisy.astype('float'))
    
    return imnoisy



if __name__=='__main__':
    
    
    n=[1,2,3,4,5,6,7,8,9,10]
    
    for i in n:
    
        path='data/sim/'
        filename=path+'raw56/'+str(i)+'.mhd'
        
        im=binRead(filename)
        im=binDilate(im.copy(), 1)
        #visStack(255*im.astype('float'))
        
        imnoisy=binNoise(im, level=.005) #level = .005, .025 ,.075 
        #visStack(255*imnoisy.astype('float'))
     
        namemat=str(i)+'.mat'
        nametif=str(i)+'.tiff'
        
        xmat={'seg':imnoisy}
        xtif=imnoisy.astype('uint8')*255
        cio.savemat(path+'data56noisy1/'+namemat, xmat, do_compression=True)
        sio.imsave(path+'data56noisy1/'+nametif, xtif)
        #visStack(imnoisy*255)