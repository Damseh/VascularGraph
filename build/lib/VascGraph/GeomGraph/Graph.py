#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 21:10:46 2019

@author: rdamseh
"""

from networkx.classes.graph import Graph as G
import networkx as nx
from copy import deepcopy

class Graph(G):
    
    def __init__(self, NodesPos=None, Edges=None, Radii=None, data=None, Types=None):        
        G.__init__(self, data=data)

        # attributes to be stored                 
        self.SetGeomGraph(NodesPos, Edges, Radii, Types)  
        self.Area=0    
        
    #private  
    def __UpdateNodesPos(self, NodesPos):
        
        AssignVals=True
        try:
            for i, p in zip(self.GetNodes(), NodesPos):
                self.node[i]['pos']=p
        except:
            AssignVals=False
            print('Cannot set \'NodesPos\'!')

    def __UpdateRadii(self, Radii):
        
        AssignVals=True
        try:
            for i, r in zip(self.GetNodes(), Radii):
                self.node[i]['r']=r
        except:
            AssignVals=False  
            print('Cannot set \'Daimeters\'!') 
 
    def __UpdateTypes(self, Types):
        
        AssignVals=True
        try:
            for i, t in zip(self.GetNodes(), Types):
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
            p=[self.node[i]['pos'] for i in self.GetNodes()]
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
    
    @property
    def EdgesIter(self):
        return iter(self.edges())
    
    ######### Radii

    def GetRadii(self):
        try:
            return [self.node[i]['d'] for i in self.GetNodes()]   
        except:
            try:
                return [self.node[i]['r'] for i in self.GetNodes()]    
            except: 
                print('No radii assigned to graph nodes!')         
                return None

    def GetTypes(self):
        try:
            return [self.node[i]['type'] for i in self.GetNodes()]   
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
        
        nodes=set()
        
        for i in bifurcation:
            u={node for node in self.GetNodes() if len(self.GetNeighbors(node))==i}
            nodes=nodes.union(u) 
            
        return list(nodes)


    def to_directed(self, as_view=False):
    
        if as_view is True:
            return nx.graphviews.DiGraphView(self)
        # deepcopy when not a view
        from VascGraph.GeomGraph import DiGraph
        G = DiGraph()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
        G.add_edges_from((u, v, deepcopy(data))
                         for u, nbrs in self._adj.items()
                         for v, data in nbrs.items())
        return G

    def LabelBranches(self):
        
        '''
        This funtion gives different id's for undirected graph branches
        -Each branch ifd is stored in 'branch' attribute of each node along that branch
        
        Input:
           graph: VascGraph.GeomGraph.Graph calss 
        '''
            
        for i in self.GetNodes():
            if len(self.GetNeighbors(i))!=2:
                self.node[i]['branch']=0
                    
        
        label=1
        c=1
        
        def propagate(self, i, label):
            
            j=i
            def forward(j_list):
                j=None
                for k in j_list:
                    try:
                        dumb=self.node[k]['branch']
                        pass
                    except:
                        j=k
                        self.node[j]['branch']=label
                        break
                return j
            
            con=1
            valid_path=False
            while con is not None:
                j_list=self.GetNeighbors(j)
                j=forward(j_list)
                if j is not None:
                    valid_path=True
                con=j
                
            return valid_path
                                  
                
        while c==1:
            pathes=0
            for i in self.GetNodes():
                if len(self.GetNeighbors(i))!=2:
                    valid_path=propagate(self, i, label)
                    if valid_path:
                        pathes+=1
                    label+=1
            if pathes==0:
                break
            
    def to_directed_branches(self): 

        '''
        transform to directed graph by:
            - splitting the graph into subgraphs (eaching containg only one branch)
            - generte directed edges on each branch
        '''
           
        bn1=self.GetJuntionNodes(bifurcation=list(range(3, 50))) # bifurcation nodes
        self.remove_nodes_from(bn1)
        
        bn2=self.GetJuntionNodes(bifurcation=[0]) # single nodes
        self.remove_nodes_from(bn2)
        
        
        subgraphs=list(nx.connected_component_subgraphs(self))
         
        startend=[]
        for gg in subgraphs:
            s=[i for i in gg.GetNodes() if len(gg.GetNeighbors(i))==1]
            startend.append(s)
        
        e=[list(nx.dijkstra_path(self, i, j)) for i, j in startend]
        edges_di=[]
        
        for i in e:
            n1=i[:-1]
            n2=i[1:]
            edges_di.append([[k1, k2] for k1, k2 in zip(n1, n2)])
            
        e=[j for i in edges_di for j in i]
        
        self.remove_edges_from(self.GetEdges())
        g = self.to_directed()
        g.add_edges_from(e) 

        return g          
            
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
    
    
    
if __name__=='__main__':
    pass
