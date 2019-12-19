#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:23:32 2019

@author: rdamseh
"""



from VascGraph.GeomGraph import Graph
from VascGraph.Tools.CalcTools import fixG
import numpy as np
import networkx as nx

class ReadCGAL:    
    
    def __init__(self, filenameVertices, filenameEdges):
        
        self.Graph=None
        self.filenameVertices=filenameVertices 
        self.filenameEdges=filenameEdges
        
        
    def getGraph(self, P1, P2, P11, P22):
    
        u1=np.unique(P1,axis=0)
        u2=np.unique(P2,axis=0)
        P_unique=np.unique(np.vstack([u1,u2]),axis=0)
        
        
        labels=dict()
        for i in range(len(P_unique)):
            labels[str(P_unique[i,:])]=i    
    
        
        intersect_ind=list()
 
        counter1=dict()
        for i in P1:
            if str(i) in counter1.keys():        
                counter1[str(i)]=counter1[str(i)]+1
            else:
                counter1[str(i)]=1   
    #    
                     
        counter2=dict()    
        for i in P2:
            if str(i) in counter2.keys():        
                counter2[str(i)]=counter2[str(i)]+1
            else:
                counter2[str(i)]=1  
    
        
        for i in labels.keys():    
            if i in counter1.keys():
                if counter1[i]>2:
                    intersect_ind.append(labels[i])
                else:
                    if counter1[i]==2 and i in counter2.keys():
                        intersect_ind.append(labels[i])
            if i in counter2.keys():  
                if counter2[i]>2:
                    intersect_ind.append(labels[i])
                else:
                    if counter2[i]==2 and i in counter1.keys():
                        intersect_ind.append(labels[i])
    
    
        intersect=P_unique[intersect_ind]    
    
           
        connections=[]
        for i in range(len(P1)):
            start=labels[str(P1[i])]
            end=labels[str(P2[i])]
            connections.append((start,end))
            
        return P_unique, intersect, np.array(connections)

        
        
    def readCGAL(self, filenameEdges, filenameVertices):
        
        f_edges = open(filenameEdges, 'r')
        c_edges=f_edges.readlines()
       
        f_verts = open(filenameVertices, 'r')
        c_verts=f_verts.readlines()
        
        def process(c):
            c=c.rstrip('\n')
            c=c.split()    
            for i in range(len(c)):
                c[i]=float(c[i])        
            return c
                
        c_edges=[process(c_edges[i]) for i in range(len(c_edges))]    
        p_edges=np.array(c_edges)
        P1=p_edges[:,0:3]
        P2=p_edges[:,3:6]
        
        c_verts=[process(c_verts[i]) for i in range(len(c_verts))]    
        p_verts=np.array(c_verts) 
        P11=p_verts[:,0:3]
        P22=p_verts[:,3:6]
                  
    
        return P1, P2, P11, P22    
        
    def Update(self, FullyCC=False):
        
        filenameVertices=self.filenameVertices 
        filenameEdges=self.filenameEdges
        
        
        P1, P2, P11, P22=self.readCGAL(filenameEdges, filenameVertices)   
        p, intersections, c=self.getGraph( P1, P2, P11, P22) 
        
        G=Graph()
        G.add_nodes_from(range(np.shape(p)[0]))
        G.add_edges_from(np.ndarray.tolist(c))
        for i in range(np.shape(p)[0]):
            G.node[i]['pos']=p[i,:]
        G.to_undirected()
        
        if FullyCC==True:
            
            # connected components
            graphs=list(nx.connected_component_subgraphs(G))
            s=0
            ind=0
            for idx, i in enumerate(graphs):
                if len(i)>s:
                    s=len(i); ind=idx
            G=graphs[ind]
            G=fixG(G) 
            
        self.Graph=G
        
    def GetOutput(self):
        
        if self.Graph is None:
            self.Update()
                
        if self.Graph is not None:
            return self.Graph        