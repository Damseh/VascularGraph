#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:23:36 2019

@author: rdamseh
"""

import networkx as nx

class WritePajek:
    
    def __init__(self, path, name, graph):

        self.path=path
        self.name=name

        graph=self.__stringizer(graph.copy())
        
        nx.write_pajek(graph, self.path+self.name)
        
    def __stringizer(self, g):
        
        test_id=g.GetNodes()[0]
        attr=['pos', 'r', 'd', 'type', 'branch', 'flow', 'pressure', 'velocity', 'po2', 'so2', 'velocity']
        attr_to_stringize=[]
        
        for i in attr:
            try:
                if not type(g.node[test_id][i]) is str:
                    attr_to_stringize.append(i)
                    
            except:
                pass

        for i in g.GetNodes():
            for j in attr_to_stringize:
                g.node[i][j]=str(g.node[i][j])
        
        
        return g
                
    def Update(self): pass