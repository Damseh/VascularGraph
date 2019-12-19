#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:59:24 2019

@author: rdamseh
"""

import networkx as nx
from VascGraph.GeomGraph import Graph, DiGraph

class ReadSWC:
    
    def __init__(self, path):
        
        self.path=path
        
        try:
            with open(self.path, 'r') as f:
                lines=f.readlines()
        except:
            print('Cannot read file!')
            
        pos=[[float(i.split(' ')[2]),
              float(i.split(' ')[3]),
              float(i.split(' ')[4])] for i in lines]
        r=[float(i.split(' ')[5]) for i in lines]

        e=[[int(i.split(' ')[6]),
            int(i.split(' ')[0])] for i in lines if int(i.split(' ')[6]) !=-1]
        
        
        self.__build_graph(e, pos, r)
        
        
    def __build_graph(self, edges, pos, radii):
        
        g=DiGraph()
        g.add_edges_from(edges)
        
        for i, p, r in zip(g.GetNodes(), pos, radii):
            g.node[i]['pos']=p
            g.node[i]['r']=r
            
        self.graph=g
        
        
    def Update(self):pass
    
    def GetOutput(self):
        return self.graph
        
        
if __name__=='__main__':
    pass       
        
        
        
        
        