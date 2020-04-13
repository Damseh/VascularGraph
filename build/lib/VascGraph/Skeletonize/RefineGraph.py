#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:03:01 2019

@author: rdamseh
"""
from VascGraph.Tools.CalcTools import *
from VascGraph.Skeletonize import BaseGraph 

class RefineGraph(BaseGraph):
    
    def __init__(self, Graph=None):
        BaseGraph.__init__(self)
        if Graph is not None:    
            self.Graph=Graph
   
    # private
    def __RefineGraph(self):
        
      while 1:
            
            self.Nodes=self.Graph.GetNodes()
            #find polys
            cyc=nx.cycle_basis(self.Graph) 
            t=[k for k in cyc if len(k)<self.PolyParam and len(k)>1] #len(k)>1 accounts for self loops
            
            #positon of poly vertices
            p=[[self.Graph.node[j]['pos'] for j in i]  for i in t]   
            
            # get polys that pass the area condition        
            ar=[CycleArea(i) for i in p]        
            t=[i for i,j in zip(t,ar) if j <self.AreaParam]
            p=[i for i,j in zip(p,ar) if j <self.AreaParam]
            
            #check if polys are found
            if len(t)==0:
                break
            
            # centers of polygons
            c=[np.mean(i, axis=0) for i in p]                   
            steps=[.5*(i-j) for i,j in zip(p,c)] 
                          
            #unravel t and p and steps        
            t=[j for i in t for j in i]
            p=[j for i in p for j in i]
            steps=[j for i in steps for j in i]
           
            #get movment for polygons vertices
            mov=dict()
            for itr, i in enumerate(t):
            
                try:
                    mov[i]=np.vstack((mov[i], steps[itr]))
                except:
                    mov[i]=steps[itr]
            
            # select random step if multiple steps exist for a vertix
            for i in mov.keys():
                
                try:
                    nm=np.shape(mov[i])[1] # check if there is more than one movments
                    ind=np.random.randint(0, nm-1)
                    mov[i]= mov[i][ind]
                
                except:
                    pass
                             
            # update nodes positions
            for i in self.Nodes:
                pos=self.Graph.node[i]['pos']
                try:
                    self.Graph.node[i]['pos']=pos-mov[i]
                except:
                    pass
            
            self.NodesToProcess=list(set(t)) # here, t is unraveled
            
            #print('Update topology [# nodes to process = '+str(len(self.NodesToProcess))+'] ...')
            
            self._BaseGraph__UpdateTopology()
        
    
    def Update(self, AreaParam=75.0, PolyParam=10,
               ClusteringResolution=1.0):
        
        self.AreaParam=AreaParam
        self.PolyParam=PolyParam
        self.ClusteringResolution=ClusteringResolution

        self.__RefineGraph()
        
    def GetOutput(self):
        self.Graph
        return self.Graph    

if __name__=='__main__':
    
    filepath='data/mri/1/AuxillaryData/VascularNetwork.tre'
    t=Tree(filepath)
    t.Update()
    graph=t.GetOutput()
    
    refine=RefineGraph(graph)
    refine.Update()
    graph=refine.GetOutput()

    visG(graph, jnodes_r=(2))























