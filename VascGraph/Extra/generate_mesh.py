#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:32:23 2018

@author: rdamseh
"""

from util_validation import *
from scipy import io as sio
import skimage.io as skio
from vtk import vtk
import scipy.ndimage as im


if __name__=="__main__":
    
    
    foldername='/home/rdamseh/IPMI paper/data/sim/data16noisy1/'
    for i in [1,2,3,4,5,6,7,8,9,10]:
        # read origional seg
        filename=str(i)
        label=sio.loadmat(foldername+filename+'.mat')['seg']
    
        # morphology processing on seg
        seg= getSegData(label)
        seg.process1(E_param=.005,AE_param=0.01, padding=False)        
        s=seg.segData.astype('uint8')  
    
        # save seg as mat after morphology processing
        s=label
        l=dict()
        l['seg']=s>0
        #sio.savemat(foldername+filename+'_morph.mat', l, do_compression=True)
    
    
        # save seg as tiff after morphology processing
        tifName=foldername+filename+'morph.tif'
        skio.imsave(tifName, s*255)
    
        # mesh file name
        stlName=foldername+filename+'.stl'
        vtkName=foldername+filename+'.vtk'
    
        #read stack
        reader = vtk.vtkTIFFReader()
        reader.SetFileName(tifName)
        reader.Update()
        
        #Threshold image
        threshold = vtk.vtkImageThreshold()
        threshold.SetInputConnection(reader.GetOutputPort())
        threshold.ThresholdByUpper(300)
        threshold.ReplaceOutOn()
        threshold.SetOutValue(0)
        threshold.ReplaceInOn()
        threshold.SetInValue(255)
        threshold.Update()
        #
        #Gaussian filter
        Gsmooth=vtk.vtkImageGaussianSmooth()
        Gsmooth.SetDimensionality(3)
        Gsmooth.SetStandardDeviation(5)
        Gsmooth.SetRadiusFactor(3)
        Gsmooth.SetInputConnection(threshold.GetOutputPort())
        Gsmooth.Update()
        #
        # Marching Contour to get mesh
        contour=vtk.vtkMarchingCubes()
        contour.SetInputConnection(threshold.GetOutputPort())
        contour.ComputeScalarsOff()
        contour.ComputeGradientsOff()
        contour.ComputeNormalsOff()
        contour.SetValue(0,2)
        contour.Update()
        #
        #smooth mesh (mesh vertices)
        smoother = vtk.vtkSmoothPolyDataFilter() 
        smoother.SetInputConnection(contour.GetOutputPort())
        smoother.SetNumberOfIterations(250)
        smoother.BoundarySmoothingOn()
        smoother.SetFeatureEdgeSmoothing(True)
        smoother.Update()
        #
        #reduce number of traingles
        deci = vtk.vtkDecimatePro()
        deci.SetInputConnection(smoother.GetOutputPort())
        deci.SetTargetReduction(0.5)
        deci.PreserveTopologyOff()
        deci.Update()
        #save
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(stlName)
        writer.SetInputConnection(contour.GetOutputPort())
        writer.SetFileTypeToBinary()
        writer.Write()

