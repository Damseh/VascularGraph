#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 21:10:46 2019

@author: rdamseh
"""

import os
import sys

# add VascGraph package to python path
try:
    sys.path.append(os.getcwd())
except: pass


import VascGraph as vg
from VascGraph.Skeletonize import GenerateGraph, ContractGraph, RefineGraph
from VascGraph.GraphLab import GraphPlot, StackPlot, MainDialogue
from VascGraph.GraphIO import ReadStackMat, ReadPajek, WritePajek
from VascGraph.Tools.CalcTools import fixG, FullyConnectedGraph
from VascGraph.Tools.VisTools import visG
import scipy.io as sio
from mayavi import mlab

if __name__=='__main__':
    

    # ------------------------------------------------------------------------#
    #parameters
    #-------------------------------------------------------------------------#
    
    #sampling: [1.0, 3.0] controls the sparsity of the initial graph
    # for 2pm, I have tested with sampling=2.0 and sampling=1.6 (gave better results) 
    # setting sampling<1.5 generates very dense graphs that were hard to contract at the end
    sampling=1.0
    
    speed_param=0.01 # speed of contraction process (smaller value-->faster dynamics)
    dist_param=0.01 # [0,1] controls contraction based on connectivitiy in graph
    med_param=1.0 # [0,1] controls contraction based on distance map (obtained from binary image)
    
    #-------------------------------------------------------------------------#
    # hyper parameters (modification on those is not suggested!)
    #-------------------------------------------------------------------------#
    #contraction
    degree_threshold=5.0 # used to check if a node is a skeletal node
    clustering_resolution=1.0 # controls the amount of graph dicimation (due to clustering) at each step
    stop_param=0.005 # controls the convergence criterion
    n_free_iteration=10 #number of iteration without checking for convergence
    
    #refinement
    area_param=50.0 # area of polygens to be decimated 
    poly_param=10 # number of nodes forming a polygon
    #-------------------------------------------------------------------------#
      
    
    #-------------------------------------------------------------------------#
    #skeletonization 
    #-------------------------------------------------------------------------#

    #load segmented angiogram
    s=ReadStackMat('synth1.mat').GetOutput()
    
    #s=ReadStackMat('data/tpm/boston/mouseVesselSegmentation.mat').GetOutput()
    #s=s[200:300,250:300,250:300]
        
    
    #s=ReadStackMat('data/tpm/seg/1.mat').GetOutput()

    
    # generate initial graph
    generate=GenerateGraph(s)
    generate.UpdateGridGraph(Sampling=sampling)
    graph=generate.GetOutput()


    
    # contract graph
    contract=ContractGraph(graph)
    contract.Update(DistParam=dist_param, 
                    MedParam=med_param, 
                    SpeedParam=speed_param, 
                    DegreeThreshold=degree_threshold, 
                    StopParam=stop_param,
                    NFreeIteration=n_free_iteration)
    gc=contract.GetOutput()
    
    
    #refine graph
    refine=RefineGraph(gc)
    refine.Update(AreaParam=area_param, 
                  PolyParam=poly_param)
    gr=refine.GetOutput()    
    
    
    #gr=FullyConnectedGraph(gr) # uncomment to get only fully connected components of the graph
    gr=fixG(gr, copy=True) # this is to fix nodes indixing to be starting from 0 (important for visualization)
    

  
    #-------------------------------------------------------------------------#
    # read/ write
    #-------------------------------------------------------------------------#

#    # save graph
    WritePajek(path='', name='mygraph.pajek', graph=fixG(gc))

#    #load graph    
    loaded_g=ReadPajek('mygraph.pajek').GetOutput()
    
    
    
    #-------------------------------------------------------------------------#
    # Visulaize
    #-------------------------------------------------------------------------#

    mlab.figure()
    stack_plot=StackPlot()
    stack_plot.Update(s)
    graph_plot=GraphPlot()
    graph_plot.Update(loaded_g)









