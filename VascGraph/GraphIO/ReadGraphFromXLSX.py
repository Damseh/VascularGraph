#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:24:41 2019

@author: rdamseh
"""



from VascGraph.GeomGraph import Graph
import numpy as np

class ReadGraphFromXLSX:


    def __init__(self, filename, graph=None):
        
        try:
            import pandas as pn
        except:
            print('To run this function, \'pandas\' sould be installed.')
            return        
        
        self.filename=filename
        self.xls=pn.read_excel(filename)
        self.Graph=graph
        
        
        
    def Update(self):

        try:
            x1, x2 = self.xls['x1'], self.xls['x2']
            y1, y2 = self.xls['y1'], self.xls['y2']
            z1, z2 = self.xls['z1'], self.xls['z2']
        
        except:
        
            x1, x2 = self.xls['V1 x'], self.xls['V2 x']
            y1, y2 = self.xls['V1 y'], self.xls['V2 y']
            z1, z2 = self.xls['V1 z'], self.xls['V2 z']    
                   
        ps=[(i,j,k) for i,j,k in zip(x1,y1,z1)] # start node
        pe=[(i,j,k) for i,j,k in zip(x2,y2,z2)] # end node
        
        # all nodes with their id's and pos's
        p=list(set(ps).union(set(pe)))
        pid=dict()
        pos=dict()
        for idx, i in enumerate(p):
            pid[str(i)]=idx
            pos[idx]=i


        # graph
        nodes=range(len(p))      
        e=[(pid[str(i)], pid[str(j)]) for i,j in zip(ps,pe)]    
        edges=[i for i in e if i[0]!=i[1]]
        
        g=Graph() 
        g.add_nodes_from(nodes)      
        g.add_edges_from(edges)

        for i in g.GetNodes():
            g.node[i]['pos']=np.array(pos[i])
            
        self.Graph=g
        
    def GetOutput(self):
        
        if self.Graph is not None:
            return self.Graph
        
        



