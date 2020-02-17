#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:51:26 2019

@author: rdamseh
"""


from networkx.classes.digraph import DiGraph as DiG
import networkx as nx
from copy import deepcopy

class DiGraph(DiG):
    
    def __init__(self, NodesPos=None, Edges=None, Radii=None, data=None, Types=None):        
        DiG.__init__(self, data)

        # attributes to be stored                 
        self.SetGeomGraph(NodesPos, Edges, Radii, Types)  
        self.Area=0            
   
    #private  
    def __UpdateNodesPos(self, NodesPos):
        
        AssignVals=True
        try:
            for i, p in zip(self.nodes().keys(), NodesPos):
                self.node[i]['pos']=p
        except:
            AssignVals=False
            print('Cannot set \'NodesPos\'!')

    def __UpdateRadii(self, Radii):
        
        AssignVals=True
        try:
            for i, r in zip(self.nodes().keys(), Radii):
                self.node[i]['r']=r
        except:
            AssignVals=False  
            print('Cannot set \'Daimeters\'!') 
 
    def __UpdateTypes(self, Types):
        
        AssignVals=True
        try:
            for i, t in zip(self.nodes().keys(), Types):
                self.node[i]['type']=t
        except:
            AssignVals=False  
            print('Cannot set \'Types\'!')  
    
    def SetGeomGraph(self, NodesPos=None, Edges=None, Radii=None, Types=None):  
        
        if NodesPos is not None:
            try:
                self.add_nodes_from(range(len(NodesPos)))
                self.__UpdateNodesPos(NodesPos)
            except: print('Cannot read \'Nodes\'!')
        
        if Edges is not None:
            try:   
                self.add_edges_from(Edges)
            except: print('Cannot read \'Edges\'!')
        
        if Radii is not None:
            self.__UpdateRadii(Radii)
        else: self.__UpdateRadii([1]*self.number_of_nodes())
        
        if Types is not None:
            self.__UpdateTypes(Types)
        else: self.__UpdateTypes([1]*self.number_of_nodes())
    

    def Fix(self):
        
        Oldnodes=self.GetNodes()
        new=range(len(Oldnodes))
        mapping={Oldnodes[i]:new[i] for i in new}
        nx.relabel_nodes(self, mapping, copy=False)    
    
    def GetNodes(self):
        n=self.nodes().keys() 
        if isinstance(n, list):
            pass
        else:
            n=list(n)
        return n    
    
    def GetNodesPos(self):
        try:
            p=[self.node[i]['pos'] for i in self.nodes().keys()]
            return p    
        except: pass

    def SetNodesPos(self, NodesPos):        
        self.__UpdateNodesPos(NodesPos) 
    
    @property
    def NodesPosIter(self):
        return iter(self.GetNodesPos())

    def GetEdges(self):    
        n=self.edges().keys() 
        if isinstance(n, list):
            pass
        else:
            n=list(n)
        return n   
    
    def GetSuccessors(self, i):
        return list(self.successors(i))
    
    def GetPredecessors(self, i): 
        return list(self.predecessors(i))
        
        
    @property
    def EdgesIter(self):
        return iter(self.edges())
    
    ######### Radii

    def GetRadii(self):
        try:
            return [self.node[i]['d'] for i in self.nodes().keys()]   
        except:
            try:
                return [self.node[i]['r'] for i in self.nodes().keys()]    
            except: 
                print('No radii assigned to graph nodes!')         
                return None

    def GetTypes(self):
        try:
            return [self.node[i]['type'] for i in self.nodes().keys()]   
        except:
                print('No types assigned to graph nodes!')         
                return None
 
    def GetFlows(self):
        try:
            return [self.node[i]['flow'] for i in self.GetNodes()]   
        except:
                print('No flows assigned to graph nodes!')         
                return None

    def GetPressures(self):
        try:
            return [self.node[i]['pressure'] for i in self.GetNodes()]   
        except:
                print('No pressures assigned to graph nodes!')         
                return None           

    def GetVelocities(self):
        try:
            return [self.node[i]['velocity'] for i in self.GetNodes()]   
        except:
                print('No velocities assigned to graph nodes!')         
                return None 
            
    def GetBranchLabels(self):
        try:
            return [self.node[i]['branch'] for i in self.nodes().keys()]   
        except:
                print('No branch labels assigned to graph nodes!')         
                return None
            
    def SetRadii(self, Radii):        
        self.__UpdateRadii(Radii)           
   
    def SetTypes(self, Types):        
        self.__UpdateTypes(Types)      
        
    @property
    def RadiiIter(self):
        try:
            return iter(self.GetRadii())  
        except: return None

    @property
    def TypesIter(self):
        try:
            return iter(self.GetTypes())  
        except: return None
 
    @property
    def BranchLabelsIter(self):
        try:
            return iter(self.GetBranchLabels())  
        except: return None
       
        ######### Neighbors
    
    def GetNeighbors(self, i=None):
        if i is None:
            return [list(self.neighbors(i)) for i in self.GetNodes()]   
        else:
            return list(self.neighbors(i))
        
    def GetNeighborsNodesPos(self):
        n= self.GetNeighbors()
        n_pos = [[self.node[i]['pos'] for i in j] for j in n]
        return n, n_pos 
    
    @property
    def NeighborsIter(self):
        return iter(self.GetNeighbors())
    
    @property
    def NeighborsNodesPosIter(self):
        n, n_pos = self.GetNeighborsNodesPos()
        return iter(n), iter(n_pos)
     
    ######### Degree

    def GetNodesDegree(self, nbunch=None, weight=None):       
        return [i[1] for i in self.degree_iter(nbunch, weight)]
    
    @property
    def NodesDegreeIter(self):       
        return iter(self.GetNodesDegree())

    ######### Calc
    
    def GetJuntionNodes(self, bifurcation=[1, 3]):
        # needs to be implemented
        pass

    def to_undirected(self, reciprocal=False, as_view=False):
            if as_view is True:
                return nx.graphviews.GraphView(self)
            # deepcopy when not a view
            from VascGraph.GeomGraph import Graph
            G = Graph()
            G.graph.update(deepcopy(self.graph))
            G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
            if reciprocal is True:
                G.add_edges_from((u, v, deepcopy(d))
                                 for u, nbrs in self._adj.items()
                                 for v, d in nbrs.items()
                                 if v in self._pred[u])
            else:
                G.add_edges_from((u, v, deepcopy(d))
                                 for u, nbrs in self._adj.items()
                                 for v, d in nbrs.items())
            return G   
        
    def ReverseEdge(self, e):
        attrs = self[e[0]][e[1]]
        self.remove_edge(e[0], e[1])
        self.add_edge(e[1], e[0])
        for k in attrs.keys():
            self[e[1]][e[0]][k]=attrs[k]

    
    def GetSourcesSinks(g):
        
        sources=[]
        sinks=[]
        
        for i in g.GetNodes():
            try:
                try:
                    if g.node[i]['source']==1:
                        sources.append(i)
                except:
                    if g.node[i]['inflow']==1:
                        sources.append(i)                
            except: pass
        
            try:
                try:
                    if g.node[i]['sink']==1:
                        sinks.append(i)
                except:
                    if g.node[i]['outflow']==1:
                        sinks.append(i)  
                        
            except: pass   
        
        return sources, sinks    
    