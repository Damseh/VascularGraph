#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:42:53 2019

@author: rdamseh
"""



import GraphPaper2018V1 


if __name__=='__main__':
    
    
    path='data/sim/data56noisy2/'
    seg=sio.loadmat(path+'1.mat')['seg']


    # testmodel #################
    l=sio.loadmat('data/test_model.mat')['model']
    l=l[:,:460,(0,5,10,20,25,30,35,40)]
    s=np.shape(l)
    s=np.array([256.0,256.0,3.0])/s
    l=sc.ndimage.zoom(l,s)
    ####################    
    
    generateGraph=GenerateGraph(seg)
    generateGraph.UpdateGridGraph(Sampling=1)
    graph=generateGraph.GetOutput()
    
    contract=ContractGraph(graph)
    contract.Update(SpeedParam=1)
    graph=contract.GetOutput()
    
    refine=RefineGraph(graph)
    refine.Update()
    graph=refine.GetOutput()

    visG(graph, jnodes_r=(2))