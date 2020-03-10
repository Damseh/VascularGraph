#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:38:36 2019

@author: rdamseh
"""
import networkx as nx 


class GraphObject:
    
    def __init__(self, Graph=None):
        
        '''
        Graph can be directed or undirected networkx graph 
        '''
        if Graph is None:
            pass
        else:
            if nx.is_directed(Graph):
                print('--Initialized graph is directed!')
            else:
                print('--Initialized graph is undirected!')
                
            self.Graph=Graph
        
    def InitGraph(self):
        
        '''
        Obtain a gaph containing only bifurcation nodes
        '''
        if self.Graph is None:
            print('Run SetGraph!')
            return
        else:
            pass
        
        self.JunctionNodes=list(self.Graph.GetJuntionNodes())  
        
        # assign  weights
        for v1, v2 in self.Graph.GetEdges():
            if (v1 in self.JunctionNodes or v2 in self.JunctionNodes):
                self.Graph[v1][v2]['weight']=1
            else:
                self.Graph[v1][v2]['weight']=0       
        
    def GetJuntionNodes(self):
                    
        nodes=self.Graph.GetJuntionNodes()
        return nodes

    def GetNotJuntionNodes(self):
                    
        return set(self.Graph.GetNodes()).difference(self.GetJuntionNodes())

    def branch_to_edges(self, p):
        p1=p[:-1]
        p2=p[1:]
        return [(i,j) for i,j in zip(p1,p2)]
        
        
    def UpdateReducedGraph(self):
        
        self.ReducedGraph=self.Graph.copy().to_undirected()

        cont=1
        while cont!=0:
            cont=0
            for i in self.ReducedGraph.GetNodes():
                k=self.ReducedGraph.GetNeighbors(i)
                if len(k)==2:
                    self.ReducedGraph.remove_node(i)
                    if (k[0] in self.JunctionNodes or k[1] in self.JunctionNodes):
                        self.ReducedGraph.add_edge(k[0], k[1], weight=1)
                    else:
                        self.ReducedGraph.add_edge(k[0], k[1], weight=0)
                    cont=1
                        
             
    def UpdateBranches(self):
        self.Branches=self.ReducedGraph.GetEdges()
        
    def UpdateDictBranches(self):
        
        try:
            self.Branches[0]
        except:
            self.UpdateBranches()
            
        nodes_branches=[list(nx.all_shortest_paths(self.Graph, 
                                   source=e[0], 
                                   target=e[1], 
                                   weight='weight')) for e in self.Branches]

        self.DictBranches=dict(zip(self.Branches, nodes_branches))                    
                    
                    
    def GetGraph(self):
        return self.Graph
    
    def SetGraph(self, Graph):
        self.Graph=Graph
        
    def GetReducedGraph(self):
        
        try:
            return self.ReducedGraph
        except: return
        
    def GetBranches(self):
        
        try:
            return self.Branches
        except: return    
        
    def GetDictBranches(self):
        
        try:
            return self.DictBranches
        except: return        

