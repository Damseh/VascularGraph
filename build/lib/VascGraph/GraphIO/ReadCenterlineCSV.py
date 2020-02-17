#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  18 09:12:47 2019

@author: rdamseh
"""



from VascGraph.Tools.CalcTools import *
from VascGraph.Tools.VisTools import *
from VascGraph.Skeletonize import RefineGraph
from VascGraph.GeomGraph import Graph

class ReadCenterlineCSV:
    '''
    Class to create a graph given only a centerline (points supposed to 
    have equal spacing between each other) 
    
    Constructer Input: CSV file with columns: X, Y, Z, Radius            
    '''
    
    def __init__(self, filepath):

        self.FilePath=filepath
        
        self.Resolution=1.0
        self.ConnectionParam=4
        
        self.__X = []
        self.__Y = []
        self.__Z = []
        self.__Radius = []

    # Private
    
    def __ReadFile(self):
    
        # read info from .tre
        with open(self.FilePath, 'r') as f:
            self.__Lines=f.readlines()
        
        X=[i.split(',')[0] for i in self.__Lines]
        self.__X=X[1:]
        Y=[i.split(',')[1] for i in self.__Lines]
        self.__Y=Y[1:]
        Z=[i.split(',')[2] for i in self.__Lines]
        self.__Z=Z[1:]
        
        Radius=[i.split(',')[3] for i in self.__Lines]
        self.__Radius=np.array(Radius[1:]).astype(float)

    
    def __ReadNodes(self):
        
        # graph nodes from centerline
        self.GraphNodes = np.array([self.__X, self.__Y, self.__Z]).T
        self.GraphNodes = self.GraphNodes.astype('float')
        self.GraphNodes, ClustersPos, Clusters= AssignToClusters(self.GraphNodes, 
                                                                 resolution=self.Resolution)
        
        self.GraphRadius=[np.max([self.__Radius[i] for i in j]) for j in Clusters]
        self.NNodes=len(self.GraphNodes)                         
            
    
    def __CreateConnections(self):      
        
        # connections from graph nodes
        self.Connections=[]
        
        length=len(self.GraphNodes)
        Tree=sp.spatial.cKDTree(self.GraphNodes)
        c=[Tree.query(i, k=self.ConnectionParam)[1] for i in self.GraphNodes]
        c=np.array(c)
         
        connections=[]
        
        for i in range(self.ConnectionParam):
            
            # obtain and fix connection from tree.query
            if i>0:
                cc=c[:,(0,i)]
                exclude=np.where(cc[:,1]>=len(self.GraphNodes))  
                cc[exclude]=0
                connections.append(cc)
            
        self.Connections=np.vstack(tuple(connections))
                    


    def __CreateGraph(self):
        
        # build graph  
        
        self.Graph=Graph()
            
        ind=np.array(range(self.NNodes))
        self.Graph.add_nodes_from(ind)  
            
        for i, p, r in zip(ind, self.GraphNodes, self.GraphRadius):
            self.Graph.node[i]['pos']=p
            self.Graph.node[i]['r']=r
            
        self.Graph.add_edges_from(self.Connections)      
        self.Graph.remove_edges_from(self.Graph.selfloop_edges())
        self.Graph=fixG(self.Graph)
     
    # public  
    def Update(self, ConnectionParam=4, Resolution=0.75):
        '''
        Update class Graph
        
        Input: 
            
            ConnectionParam: control number of closest neighbors 
                                     to a centreline point.
                                     
            Resolution: control at which resolution centerline 
                                points should sampled.
                                Higher value imposes lower sampling rate. 
                                0<'Resolution'<=1

        Output: create NetworkX undirected graph
        '''
        self.ConnectionParam = ConnectionParam
        self.Resolution = Resolution
        self.__ReadFile()
        self.__ReadNodes()
        self.__CreateConnections()
        self.__CreateGraph()
    
    def GetOutput(self):
        
        refine=RefineGraph(self.Graph)
        refine.Update()
        self.Graph=refine.GetOutput()
        return self.Graph
  

if __name__=='__main__':
    
    
    filepath='/home/rdamseh/GraphPaper2018V1/data/raa/models/C0001/morphology/centerlines.csv'
    g=ReadCenterlineCSV(filepath)
    g.Update(ConnectionParam=5, Resolution=.5)
    graph=g.GetOutput()
    
    if nx.number_connected_components(graph):
        pass
    else: print('Graph is not connected')
                
    visG(graph, radius=.1, gylph_r=.5, jnodes_r=.5, jnodes_c= (0,.7,.7), diam=True)

    from VascGraph.GraphLab import ModifyGraph
    
    g=fixG(reduceG(graph.copy()))
    m=ModifyGraph(g)
    m.configure_traits()


    nx.ma