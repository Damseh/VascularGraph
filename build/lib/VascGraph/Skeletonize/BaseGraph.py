#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 12:14:48 2019

@author: rdamseh
"""


from VascGraph.Tools.CalcTools import *

class BaseGraph:
    
    def __init__(self, label_ext=False):
        
        self.label_ext=label_ext


    def __UpdateTopology(self, resolution=1.0):
        
        '''
        update topology of the graph after 
        a contraction step
        '''
      
        Pos=np.array([self.Graph.node[i]['pos'] for i in self.NodesToProcess])
        
        # cluster graph nodes  
        if Pos.shape[0]>0:
            
            Centroids, ClustersPos, Clusters= AssignToClusters(Pos, resolution=resolution) 
            
            #map indicies in Clusters to that in graph
            Clusters=[[self.NodesToProcess[j] for j in i]  for i in Clusters]
            
            NClusters=len(Centroids)      
            
            if NClusters>0:
                
                # update graph with new connections based on the new nodes              
                self.__ConnectionSurgery(Centroids=Centroids,
                                         Clusters=Clusters)

    def __ConnectionSurgery(self, Clusters, Centroids=None):       
            
        '''
        Perform topological surgery based on the new clusters
        
        Input: 
            graph: networkx undirected graph
            centroids: cluters' centeriods 
            nodes: indices of graph nodes for each cluster
            voidNodes: nodes to be exludeed from the process       
        '''
        
        def GetNbrsOfNbrs(graph, NewNodes, Clusters): 
            
            NbrsOfNbrs=[]  
             
            for ind, i in enumerate(NewNodes): 
                
                # get neighbours of veticies belonging to this cluster
                nbrs=list(Clusters[ind])
                
                #obtain neighbors of neighbors
                nbrsOfnbrs=[graph.GetNeighbors(j) for j in nbrs]                      
                
                # flatten and extract a set (no duplicates)
                nbrsOfnbrs=[j for k in nbrsOfnbrs for j in k]
                nbrsOfnbrs=list(set(nbrsOfnbrs).difference(set(nbrs)))
                NbrsOfNbrs.append(nbrsOfnbrs)  
                
            return NbrsOfNbrs
            

        # add new nodes to graph
        NewNodes=list(1+np.array(range(len(Centroids)))+np.max(self.Graph.GetNodes()))
        self.Graph.add_nodes_from(NewNodes)
    
        # obtain and set Centroids if not found
        if not Centroids:
            ClustersPos=[[self.Graph.node[i]['pos'] for i in j] for j in Clusters]
            Centroids=[np.mean(np.array(i), axis=0) for i in ClustersPos]
        for ind, i in enumerate(NewNodes):
            self.Graph.node[i]['pos']=Centroids[ind]
     
    
        # obtain and set diamters of new nodes
        try:
            NewDiameters=[np.max([ self.Graph.node[j]['d'] for j in i ]) for i in Clusters]
            for ind, i in enumerate(NewNodes):
                self.Graph.node[i]['d']=NewDiameters[ind]
        except:
            pass  
        
        try:
            NewDiameters=[np.max([ self.Graph.node[j]['r'] for j in i ]) for i in Clusters]
            for ind, i in enumerate(NewNodes):
                self.Graph.node[i]['r']=NewDiameters[ind]
        except:
            pass 
        
        if self.label_ext:
            try:
                for ind, i in enumerate(NewNodes):
                    self.Graph.node[i]['ext']=0
            except:
                pass 

        
        # obtain void nodes
        ClusteredNodes=[j for i in Clusters for j in i] # unravel
        VoidNodes=list(set(self.Nodes).difference(set(ClusteredNodes)))        
                    
        ##### Add connection 1 ##### 
        
        #get neighbours of neighbourse for points in clusters
        NbrsOfNbrs=GetNbrsOfNbrs(self.Graph, NewNodes, Clusters)
         
        #build new conenctions and assign to the graph    
        NewConnections1=[[i,j] 
                        for ind, i in enumerate(NewNodes) 
                        for j in NbrsOfNbrs[ind]]
        self.Graph.add_edges_from(NewConnections1) # assign connections 1              
    
        ##### Add connection 2 ##### 
    
        NodesToKeep=set(NewNodes).union(set(VoidNodes))
        OtherNodes=list(set(self.Nodes).difference(NodesToKeep))              
        
        #map 'OtherNodes' with their corresponding cluster
        OtherNodesCluster=dict(zip(OtherNodes, [0 for i in OtherNodes]))
        for c, i in zip(NewNodes, Clusters):
            for j in i:
                OtherNodesCluster[j]=c  
                
        #find clusters with direct connection between their nodes  
        NewConnections2=[]
        for c, i in zip(NewNodes, NbrsOfNbrs):
            for j in i:
                try:
                    NewConnections2.append([c, OtherNodesCluster[j]])                  
                except:
                    pass
        self.Graph.add_edges_from(NewConnections2)     
    
        # remove nodes         
        self.Graph.remove_nodes_from(OtherNodes)










