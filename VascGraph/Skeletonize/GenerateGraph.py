#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 11:03:53 2019

@author: rdamseh
"""

from VascGraph.Tools.CalcTools import *
from VascGraph.GeomGraph import Graph
import scipy.ndimage as image
from time import time

from scipy.ndimage import filters as filt


class GenerateGraph:
    
    def __init__(self, Label):
        
        self.Label=Label
        self.Shape=np.shape(self.Label) # size of image
        self.Length=self.Shape[0]*self.Shape[1]*self.Shape[2] # number of voxels
        self.__ComputeArea()
    # private 

    def __ComputeArea(self):
        
        self.Area=np.sum(self.Label>0)
    
    def __CalculateDistMap(self):
#
#        XY=[self.Label[i,:,:] for i in range(self.Shape[0])] #Z-XY
#        ZX=[self.Label[:,:,i] for i in range(self.Shape[2])] #Y-ZX 
#        ZY=[self.Label[:,i,:] for i in range(self.Shape[1])] #X-ZY
#        
#        DistXY=np.array([image.morphology.distance_transform_edt(i) for i in XY])
#        DistZX=np.array([image.morphology.distance_transform_edt(i) for i in ZX])
#        DistZY=np.array([image.morphology.distance_transform_edt(i) for i in ZY])
#        
#        DistZX=np.rollaxis(DistZX, 0, 3)
#        DistZY=np.rollaxis(DistZY, 0, 2)      
#        
#        DistMap_=np.maximum(DistXY, DistZX) 
#        DistMap=np.maximum(DistMap_, DistZY)
#    
#        DistMap=filt.maximum_filter(DistMap, size=(3,3,3))
        
        DistMap=image.morphology.distance_transform_edt(self.Label)
        self.DistMap=DistMap
    
    def __AssignDistMapToGraph(self):
        
        '''
        Assign dist values to graph nodes 
        '''
        Nodes=self.Graph.GetNodes()
        
        for i in Nodes:
            
            Pos=tuple(self.Graph.node[i]['pos'].astype(int))
            
            if Pos[0]<self.Shape[0] and Pos[1]<self.Shape[1] and Pos[2]<self.Shape[2]:           
                
                Dist=self.DistMap[Pos]               
                
                if Dist<1: 
                    Dist=1
                    
                self.Graph.node[i]['r']=Dist
                
            else:
                self.Graph.node[i]['r']=1
        
        
    def __GenerateRandomGraphFromLabel(self):

        #random sampling
        x=np.random.uniform(low=0, high=self.Shape[0], size=self.NInitialNodes).tolist()
        y=np.random.uniform(low=0, high=self.Shape[1], size=self.NInitialNodes).tolist()
        z=np.random.uniform(low=0, high=self.Shape[2], size=self.NInitialNodes).tolist()          
        
        NodesIndices=self.Label[(np.floor(x).astype('int'),
                                 np.floor(y).astype('int'),
                                 np.floor(z).astype('int'))]>0
        
        Index=np.array([x,y,z]).T
        NodesPos=Index[NodesIndices]

        # build graph 
        self.NNodes=len(NodesPos)
        self.Graph=Graph()
        self.Graph.add_nodes_from(range(self.NNodes))
        
        # assign positions to nodes      
        for i, p in zip(self.Graph.GetNodes(), NodesPos):
            self.Graph.node[i]['pos']=p
               
        # build connectivity 
        Tree = sp.spatial.cKDTree(NodesPos)
        NeigborsIndices=Tree.query(NodesPos, k=self.Connection+1)[1]   
        
        Edges=[]
        for ind, i in enumerate(NeigborsIndices):
            Neigbours=np.unique(i)
            c=[[ind, j]  for j in Neigbours if j != ind and j != self.NNodes]            
            if c:
                Edges.append(c)
        
        #assign connections         
        Edges=[j for i in Edges for j in i] # unravel
        self.Graph.add_edges_from(Edges)  


    
    def __GenerateRandomGridGraphFromLabel(self):

        IndexTrueVoxels=np.where(self.Label)
        Index=np.array(IndexTrueVoxels).T
        
        #Limit NNodes to # of ture voxels
        if self.NNodes>len(Index): self.NNodes=len(Index)
       
        # probibility of true voxels
        Probability=(self.Label).astype(float)/np.sum(self.Label)
        Probability=Probability[IndexTrueVoxels]
        
        # obtain nodes
        NodesIndices=np.random.choice(range(len(Probability)), self.NNodes, p=Probability)
        NodesPos=Index[NodesIndices]       
        
               
        # build graph 
        self.Graph=Graph()
        self.Graph.add_nodes_from(range(self.NNodes))
        
        # assign positions to nodes      
        for i, p in zip(self.Graph.GetNodes(), NodesPos):
            self.Graph.node[i]['pos']=p
               
        # build connectivity 
        Tree = sp.spatial.cKDTree(NodesPos)
        NeigborsIndices=Tree.query(NodesPos, k=self.Connection+1)[1]   
        
        Edges=[]
        for ind, i in enumerate(NeigborsIndices):
            Neigbours=np.unique(i)
            c=[[ind, j]  for j in Neigbours if j != ind and j != self.NNodes]            
            if c:
                Edges.append(c)
        
        #assign connections         
        Edges=[j for i in Edges for j in i] # unravel
        self.Graph.add_edges_from(Edges)  

    def __GenerateGridGraphFromLabel(self):
        
        def VoxelsPositions(Label, Shape, Length):
    
            '''
            Shape: shape of array     
            indexing in order: rows by row->depth
            '''
            # positions of each voxel
            z,x,y=np.meshgrid(range(Shape[0]),
                                  range(Shape[1]),
                                  range(Shape[2]), indexing='ij')
            x=x[Label.astype(bool)]
            y=y[Label.astype(bool)]
            z=z[Label.astype(bool)]
            VoxelsPos=np.transpose([z,x,y])  
            
            return VoxelsPos
        
        def GetConnections(Label, Shape, Length):
            
            # connections from pathways on array grid
            Array=(np.reshape(range(Length), Shape)+1)*Label   
            
            # incides of voxels in the Array
            VoxelsIndices = Array[Label.astype(bool)]
            
            #--------    
            path1=iter(np.transpose([Array[:,:,0:-1].ravel(), 
                                     Array[:,:,1:].ravel()]))
            path1=(i for i in path1 if all(i))
            #--------
            path2=iter(np.transpose([np.swapaxes(Array[:,0:-1,:],1,2).ravel(), 
                                     np.swapaxes(Array[:,1:,:],1,2).ravel()]))
            path2=(i for i in path2 if all(i))
            #--------
            path3=iter(np.transpose([np.swapaxes(Array[0:-1,:,:],0,2).ravel(), 
                                     np.swapaxes(Array[1:,:,:],0,2).ravel()]))
            path3=(i for i in path3 if all(i))
        
            return VoxelsIndices, path1, path2, path3
        
        if self.Sampling is not None:          
            Scale=(1.0/self.Sampling, 1.0/self.Sampling, 1.0/self.Sampling)    
            Label=image.zoom(self.Label.astype(int), Scale)
            Shape=np.shape(Label) # size of image
            Length=Shape[0]*Shape[1]*Shape[2] # number of voxels  
        else:
            Label=self.Label
            Shape=self.Shape
            Length=self.Length
        
        # voxel indices and thier positions
        t1=time()
        VoxelsPos=VoxelsPositions(Label, Shape, Length)
        print('create nodes: '+str(time()-t1))
        
        t1=time()
        VoxelsIndices, Connections1, Connections2, Connections3=GetConnections(Label, Shape, Length)
        print('create connections: '+str(time()-t1))
                
        # build graph      
        t1=time()
        self.Graph=Graph()
        self.Graph.add_nodes_from(VoxelsIndices)
        for ind, p in zip(VoxelsIndices, VoxelsPos):
            self.Graph.node[ind]['pos']=p
                
        self.Graph.add_edges_from(Connections1)
        self.Graph.add_edges_from(Connections2)
        self.Graph.add_edges_from(Connections3)

        #exclude nodes with less than 2 neighbors
        NNodesToExclude=1
        while NNodesToExclude>0:
            NodesToExclude=[i for i in self.Graph.GetNodes() if len(self.Graph.GetNeighbors(i))<=2]
            self.Graph.remove_nodes_from(NodesToExclude)
            NNodesToExclude=len(NodesToExclude)
    
        if self.Sampling is not None:
            for i in self.Graph.GetNodes():
                self.Graph.node[i]['pos']=self.Graph.node[i]['pos']*self.Sampling
        
        print('create graph: '+str(time()-t1))
      
    # public    
    def UpdateRandomGraph(self, connection=8, nInitialNodes=100000):
        self.Connection=connection
        self.NInitialNodes=nInitialNodes
        self.__GenerateRandomGraphFromLabel()
        
    def UpdateRandomGridGraph(self, connection=8, nNodes=100000):
        self.Connection=connection
        self.NNodes=nNodes
        self.__GenerateRandomGridGraphFromLabel()
        
    def UpdateGridGraph(self, Sampling=None):
        if Sampling is not None:            
            self.Sampling=float(Sampling)
        else:
            self.Sampling=Sampling
        self.__GenerateGridGraphFromLabel()
            
            
    def GetOutput(self):
        self.__CalculateDistMap()
        self.__AssignDistMapToGraph()
        self.Graph=fixG(self.Graph)
        self.Graph.Area=self.Area
        return self.Graph
    
    def GetArea(self): return self.Area
    
    def GetDistMap(self):
        self.__CalculateDistMap()
        return self.DistMap   
    
    
if __name__=='__main__':
    
    import scipy.io as sio
    import scipy as sc
    
    path='/home/rdamseh/GraphPaper2018V1/data/sim/data56noisy2/'
    seg=sio.loadmat(path+'1.mat')['seg']


    # testmodel #################
    l=sio.loadmat('/home/rdamseh/GraphPaper2018V1/data/test_model.mat')['model']
    l=l[:,:460,(0,5,10,20,25,30,35,40)]
    s=np.shape(l)
    s=np.array([256.0,256.0,3.0])/s
    l=sc.ndimage.zoom(l,s)
    ####################    
    
    generateGraph=GenerateGraph(seg)
    generateGraph.UpdateGridGraph(Sampling=1)
    graph=generateGraph.GetOutput()
    
    
    
    visG(graph, diam=True)
    
    
    
    
    