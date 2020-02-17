#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 12:34:57 2019

@author: rdamseh
"""

from VascGraph.Tools.CalcTools import *
from VascGraph.Tools.VisTools import *
from VascGraph.GeomGraph import GraphObject
from VascGraph.Skeletonize.BaseGraph import *

class RefineGraphRadius(BaseGraph):
    
    def __init__(self, Graph=None):
        
        if Graph is not None:    
            self.Graph=Graph
           
    # private
    def __RefineRadius(self, Cutoff=3, Mode='mean'):

        if self.Graph.is_directed():
            return
        
        # obtain neighbours of cutoff degree
        nbrs=[list(nx.bfs_edges(self.Graph, i, 
                                depth_limit=Cutoff)) for i in self.Graph.GetNodes()]   
     
        nbrs=[np.ravel(i).tolist() for i in nbrs]
        nbrs=[list(set(i)) for i in nbrs]

        # 
        try:
            radii=[[self.Graph.node[j]['r'] for j in i] for i in nbrs]
        except:
            try: radii=[[self.Graph.node[j]['d'] for j in i] for i in nbrs]
            except:return
            
        if Mode=='median':
            radii=[np.median(i) for i in radii]
        
        elif Mode=='max':
            radii=[np.max(i) for i in radii]

        elif Mode=='min':
            radii=[np.min(i) for i in radii]
        else:                
            radii=[np.mean(i) for i in radii]
        
        #
        for i, r in zip(self.Graph.GetNodes(), radii):
            self.Graph.node[i]['r']=r
        
 
    def __RefineRadiusDirected(self, Cutoff=3, Mode='mean'):

        if not self.Graph.is_directed():
            return
        
        nbrs=[]
        
        
        for i in self.Graph.GetNodes():
            
            forward=list(nx.bfs_edges(self.Graph, i, reverse=False, depth_limit=Cutoff))
            if len(forward)>0:
                forward=[k for j in forward for k in j]
            backward=list(nx.bfs_edges(self.Graph, i, reverse=True, depth_limit=Cutoff))
            if len(backward)>0:
                backward=[k for j in backward for k in j]
            nbrs.append(list(set(forward).union(set(backward))))
        
        # 
        try:
            radii=[[self.Graph.node[j]['r'] for j in i] for i in nbrs]
        except:
            try: radii=[[self.Graph.node[j]['d'] for j in i] for i in nbrs]
            except:return
            
        if Mode=='median':
            radii=[np.median(i) for i in radii]
        
        elif Mode=='max':
            radii=[np.max(i) for i in radii]

        elif Mode=='min':
            radii=[np.min(i) for i in radii]
        else:                
            radii=[np.mean(i) for i in radii]
        
        #
        for i, r in zip(self.Graph.GetNodes(), radii):
            self.Graph.node[i]['r']=r
            

    def __FixedRadiusOnBranches(self, Mode='max', DictDirectedBranches=None):
        
        if DictDirectedBranches==None:
            return
        
        for key in DictDirectedBranches.keys():
            
            i=DictDirectedBranches[key]
            nodes=[k for j in i for k in j[1:]]
            radii=[self.Graph.node[k]['r'] for k in nodes]
            
            if Mode=='mean':
                r=np.mean(radii)
                
            elif Mode=='median':
                r=np.median(radii)
                
            elif Mode=='min':
                r=np.min(radii)
                
            else:
                r=np.max(radii)
            
            for k in nodes:
                self.Graph.node[k]['r']=r         
                       
    def UpdateRefineRadius(self, Cutoff=3, Mode='mean'):
        self.__RefineRadius(Mode=Mode, Cutoff=Cutoff)
        
    def UpdateRefineRadiusDirected(self, Cutoff=3, Mode='mean'):
        self.__RefineRadiusDirected(Mode=Mode, Cutoff=Cutoff)
    
    def UpdateFixedRadiusOnBranches(self, Mode='max', DictDirectedBranches=None):
        self.__FixedRadiusOnBranches(Mode=Mode, 
                                     DictDirectedBranches=DictDirectedBranches)
        
        
    def GetOutput(self):
        return self.Graph

      

if __name__=='__main__':
    
    filepath='/home/rdamseh/GraphPaper2018V1/data/mra/2/AuxillaryData/VascularNetwork.tre'
    t=ReadMRIGraph(filepath)
    t.Update()
    graph=t.GetOutput()

    refine=RefineGraphRadius(graph.copy())
    refine.Update(Cutoff=100, Mode='median')
    g=refine.GetOutput()
    
    mlab.figure()
    visG(graph, diam=True)   
    mlab.figure()
    visG(g, diam=True)


















