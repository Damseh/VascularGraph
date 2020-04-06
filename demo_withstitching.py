#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:36:33 2020

@author: rdamseh
"""




from VascGraph.Skeletonize import Skeleton
from VascGraph.GraphIO import ReadStackMat
from VascGraph.GraphLab import StackPlot
from VascGraph.Tools.VisTools import visG

if __name__=='__main__':

    '''
    This demo explains how graphing of scalable inputs can be done through patch-based graphig and 	stiching 		
    '''
    s=ReadStackMat('synth2.mat').GetOutput()

    #contraction
    speed_param=0.05
    dist_param=0.0
    med_param=1.0
    degree_threshold=10.0 # used to check if a node is a skeletal node
    sampling=1
    clustering_r=1

    #contraction
    stop_param=0.002 # controls the convergence criterion
    n_free_iteration=3 #number of iteration without checking for convergence
    
    #refinement
    area_param=50.0 # area of polygens to be decimated 
    poly_param=10 # number of nodes forming a polygon    
    
    #update with stiching
    size=100
    is_parallel=True
    n_parallel=5
    niter1=10 
    niter2=5 
    
    sk=Skeleton(label=s, 
                speed_param=speed_param,
                dist_param=dist_param,
                med_param=med_param,
                sampling=sampling,
                degree_threshold=degree_threshold,
                clustering_resolution=clustering_r,
                stop_param=stop_param,
                n_free_iteration=n_free_iteration,
                area_param=area_param,
                poly_param=poly_param)
    
    sk.UpdateWithStitching(size=size,
                           niter1=niter1, 
                           niter2=niter2,
                           is_parallel=is_parallel, 
                           n_parallel=n_parallel)
    
    fullgraph=sk.GetOutput()

    print('--Visualize final skeleton ...')
    splot = StackPlot(new_engine=True)    
    splot.Update((s>0).astype(int))
    visG(fullgraph)






