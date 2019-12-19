#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 09:12:47 2019

@author: rdamseh
"""



from VascGraph.Tools.CalcTools import *
from VascGraph.Tools.VisTools import *
from VascGraph.Skeletonize import RefineGraph
from VascGraph.GeomGraph import Graph

class ReadMRIGraph:
    
    ''' Read 'tre' file for graphs genrated from MRI images'''
    
    def __init__(self, filepath):

        self.FilePath=filepath

        self.__ObjectType = []
        self.__ObjectSubType = []
        self.__NDims = []
        self.__ID = []
        self.__ParentID = []
        self.__Color = []
        self.__TransformMatrix = []
        self.__Offset = []
        self.__CenterOfRotation = []
        self.__ElementSpacing = []
        self.__Root = []
        self.__Artery = []
        self.__PointDim = []
        self.__NPoints = []
        self.__Points = []

        self.__StartObjectIndices=[]
        self.__StartPointsIndices=[]
        

    # Private
    
    def __ReadFile(self):
    
        # read info from .tre
        with open(self.FilePath, 'r') as f:
            self.__Lines=f.readlines()

        Read=False 
        
        for idx, line in enumerate(self.__Lines):        
        
            if line.split()[0]=='ObjectType' and line.split()[2]=='Tube':
                
                Read=True
                self.__StartObjectIndices.append(idx)
                self.__ObjectType.append(line.split()[2:]) 
            
            else:
                
                if line.split()[0]=='ObjectType' and line.split()[2]!='Tube':
                    Read=False
        
            if line.split()[0]=='ObjectSubType' and Read:
                self.__ObjectSubType.append(line.split()[2:]) 
        
            if line.split()[0]=='NDims' and Read:
                self.__NDims.append(line.split()[2:]) 
        
            if line.split()[0]=='ID' and Read:
                self.__ID.append(line.split()[2:]) 
        
            if line.split()[0]=='ParentID' and Read:
                self.__ParentID.append(line.split()[2:]) 
        
            if line.split()[0]=='Color' and Read:
                self.__Color.append(line.split()[2:]) 
        
            if line.split()[0]=='TransformMatrix' and Read:
                self.__TransformMatrix.append(line.split()[2:]) 
        
            if line.split()[0]=='Offset' and Read:
                self.__Offset.append(line.split()[2:]) 
        
            if line.split()[0]=='CenterOfRotation' and Read:
                self.__CenterOfRotation.append(line.split()[2:]) 
        
            if line.split()[0]=='ElementSpacing' and Read:
                self.__ElementSpacing.append(line.split()[2:]) 
        
            if line.split()[0]=='Root' and Read:
                self.__Root.append(line.split()[2:]) 
                
            if line.split()[0]=='Artery' and Read:
                self.__Artery.append(line.split()[2:]) 
                
            if line.split()[0]=='PointDim' and Read:
                self.__PointDim.append(line.split()[2:]) 
                
            if line.split()[0]=='NPoints' and Read:
                self.__NPoints.append(line.split()[2:])
                
            if line.split()[0]=='Points' and Read:
                self.__StartPointsIndices.append(idx+1) 
        
        # number of points in each segment
        self.__NPoints=np.array(self.__NPoints).astype(int)
        self.__NPoints=self.__NPoints.ravel()


    def __ReadSegments(self):
        
        #read segments (tube objects)
        self.Segments=[]
        self.SegmentsRadii=[]
        
        for start, npoints in zip(self.__StartPointsIndices, self.__NPoints):       
            s=self.__Lines[start:start+int(npoints)]
            s=[i.split() for i in s]
            s=np.array(s).astype(float)
            self.SegmentsRadii.append(s[:,3])
            self.Segments.append(s[:,(0,1,2)])
        

    def __ReadNodes(self):
        
        # nodes from segments
        self.SegmentsNodes=[]
        self.SegmentsNodesRadii=[]
        self.NNodes=[]
        for i, r in zip(self.Segments, self.SegmentsRadii):
            nodes ,_ ,ids = AssignToClusters(i)
            radii=[np.max(r[k]) for k in ids]
            self.SegmentsNodes.append(nodes)
            self.SegmentsNodesRadii.append(radii)
            self.NNodes.append(len(nodes))

    def __CreateConnections(self):
        
        # connections from segments
        self.SegmentsConnections=[]
        
        for segment in self.SegmentsNodes:   
            length=len(segment)
            Tree=sp.spatial.cKDTree(segment)
            c=[Tree.query(i, k=3, distance_upper_bound=2.0)[1] for i in segment]
            c=np.array(c)
            
            # obtain and fix connection from tree.query
            c1=c[:,(0,1)]
            exclude1=np.where(c1[:,1]>=len(segment))  
            c1[exclude1]=0
            
            c2=c[:,(0,2)]
            exclude2=np.where(c2[:,1]>=len(segment))   
            c2[exclude2]=0
        
            c=np.vstack((c1,c2))
            
            self.SegmentsConnections.append(c)

    def __CreateGraph(self):
        
        # build graph  
        self.Graph=Graph()
        totalnodes=0
        
        for nodes, c, radii, n in zip(self.SegmentsNodes, 
                                      self.SegmentsConnections, 
                                      self.SegmentsNodesRadii, 
                                      self.NNodes):
            
            ind=np.array(range(n))+totalnodes
            self.Graph.add_nodes_from(ind)  
            
            for i, p, r in zip(ind, nodes, radii):
                self.Graph.node[i]['pos']=p
                self.Graph.node[i]['r']=r
                
            self.Graph.add_edges_from(c+totalnodes)      
            totalnodes+=n
        self.Graph.remove_edges_from(self.Graph.selfloop_edges())
        self.Graph=fixG(self.Graph)

      
    # public
    
    def Update(self):
        
        self.__ReadFile()
        self.__ReadSegments()
        self.__ReadNodes()
        self.__CreateConnections()
        self.__CreateGraph()
        
    def GetSegmentsNodes(self):
        return self.SegmentsNodes
    
    def GetSegmentsNodesRadii(self):
        return self.SegmentsNodesRadii

    def GetSegmentsConnections(self):
        return self.SegmentsConnections
    
    def GetOutput(self):
        refine=RefineGraph(self.Graph)
        refine.Update()
        self.Graph=refine.GetOutput()
        return self.Graph
        
if __name__=='__main__':
    
    filepath='/home/rdamseh/GraphPaper2018V1/data/mra/2/AuxillaryData/VascularNetwork.tre'
    t=ReadMRIGraph(filepath)
    t.Update()
    g=t.GetOutput()
    
    from mayavi import mlab
    mlab.figure()
    visG(g, jnodes_r=1, jnodes_c= (0,.7,.7), diam=True)


