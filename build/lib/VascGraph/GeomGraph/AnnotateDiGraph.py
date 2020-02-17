#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:56:17 2019

@author: rdamseh
"""

from VascGraph.GeomGraph import DiGraph
from VascGraph.GraphLab import GraphPlot, StackPlot
from VascGraph.GraphIO import ReadPajek, ReadStackMat
from VascGraph.Tools.CalcTools import *

from mayavi import mlab
from VascGraph.Tools.VisTools import visG

from VascGraph.GeomGraph.GenerateDiGraph import GenerateDiGraph
import numpy as np 

class AnnotateDiGraph(GenerateDiGraph):
    
    def __init__(self, Graph=None, Sources=None):
        GenerateDiGraph.__init__(self, Graph)
        
        if Sources is not None:
            self.Sources=Sources
            self.UpdateDiGraphFromGraph(Sources=self.Sources)
                   
    def ResetTypeAttr(self):
        
        for i in self.DiGraph.GetNodes():
            self.DiGraph.node[i]['type']=1   

    def Reverse(self):
        
        try:
            self.DiGraph
        except:
            print('UpdateDiGraph!')
            return 
        
        self.DiGraph=self.DiGraph.reverse()
        self.ReducedDiGraph=self.ReducedDiGraph.reverse()
        
        keys=self.DictDirectedBranches.keys()
        keys_=[(i[1],i[0]) for i in keys]
        values=[self.DictDirectedBranches[i] for i in keys]
        self.DictDirectedBranches=dict(zip(keys_,values))
        
        
    def PropagateTypes(self, Starting_nodes=[], cutoff=1, value=1, backward=False):
        
        '''
        input:
            t: type to be assigned to graph nodes
        '''
        if self.DiGraph is None:
            print('Run UpdateDiGraph!')
            return
        
        if len(Starting_nodes)==0:
            if backward==False:
                print('Sources need to be set!')
            elif backward==True:
                print('Sinks need to be set!')
            raise ValueError
    
        def get_nodes_from_propagation(source, cutoff, backward):

            search=list(nx.bfs_edges(self.ReducedDiGraph, 
                                      source, 
                                      reverse=backward, 
                                      depth_limit=cutoff))
            nodes=[k for j in search for k in j]
            nodes=list(set(nodes))
            branches=[e for e in self.ReducedDiGraph.edges() if (e[0] in nodes and e[1] in nodes)]
            pathes=[self.DictDirectedBranches[b] for b in branches]
            return list(set([k for i in pathes for j in i for k in j]))
        
        nodes=[]
        for source in Starting_nodes:
            nodes.append(get_nodes_from_propagation(source, cutoff, backward))
        nodes=list(set([j for i in nodes for j in i]))
        
        for i in nodes:
            self.DiGraph.node[i]['type']=value



    def PropagateTypes2(self, cutoff=1, value=1, exclude_values=[2], other_value=3, backward=False):
            
            '''
            input:
                t: type to be assigned to graph nodes
            '''
            if self.DiGraph is None:
                print('Run UpdateDiGraph!')
                return
            
            try:
                self.DiGraph.node[self.DiGraph.GetNodes()[0]]['branch']
            except:
                print('Digraph should have \'branch\' labels! \n Ex. g.node[id][\'branch\']=1')
                return

            if backward:
                max_b=np.max([self.DiGraph.node[i]['branch'] for i in self.DiGraph.GetNodes()])
                branches=np.arange(max_b, max_b-cutoff, -1)
                
            else:
                min_b=np.min([self.DiGraph.node[i]['branch'] for i in self.DiGraph.GetNodes()])
                branches=np.arange(min_b, min_b+cutoff, 1)
            
            for b in branches:
                for i in self.DiGraph.GetNodes():
                    
                    if self.DiGraph.node[i]['branch']==b:
                        self.DiGraph.node[i]['type']=value
                        
                    elif self.DiGraph.node[i]['type'] not in exclude_values:
                        self.DiGraph.node[i]['type']=other_value
            
                    else: pass



    def PropagateCapillaryTypes(self, value=3, values_to_exlude=[1,2,3]):
        
        for i in self.DiGraph.GetNodes():
            if self.DiGraph.node[i]['type'] not in values_to_exlude:
                self.DiGraph.node[i]['type']=value
            
        
    def CloseEnds(self, EndNodes=[]):
        
        try: 
            self.DiGraph
        except: return
        
        ReducedDiGraph=self.ReducedDiGraph.copy()
        
        def get_nodes_from_branches(branches):
            # edges
            edges=[self.DictDirectedBranches[i] for i in branches]
            # nodes
            return list(set([k for i in edges for j in i for k in j[1:]])) # avoid source node on that branch
        
        edges_before=ReducedDiGraph.GetEdges()
        
        nodes_to_exclude=EndNodes
        count=1
        while count>0:
            for i in ReducedDiGraph.GetNodes():
                count=0
                p=ReducedDiGraph.GetPredecessors(i) 
                s=ReducedDiGraph.GetSuccessors(i) 
                if (len(p)>0 and len(s)==0 and i not in nodes_to_exclude):
                    ReducedDiGraph.remove_node(i)
                    count+=1
        
        branches_to_remove=set(edges_before).symmetric_difference(set(ReducedDiGraph.GetEdges()))
        nodes_to_remove=get_nodes_from_branches(branches_to_remove)  
        self.DiGraph.remove_nodes_from(nodes_to_remove)
            
    def GenerateColors(self, length):
        
        r=np.arange(0.0,1,.075)
        g=np.arange(0.0,1,.075)
        b=np.arange(0.0,1,.075) 
        colors=np.meshgrid(r,g,b)
        colors=np.array([np.ravel(colors[0]), np.ravel(colors[1]), np.ravel(colors[2])]).T
        return np.flip(colors[:length], 0)

    
    def LabelDiGraphBranching(self, sources=[]):
        
        try: self.DiGraph
        except:
            print('UpdateDiGraph')
            return 
        
        if len(sources)==0:
            sources=[]
            for i in self.DiGraph.GetNodes():
                try:
                    if self.DiGraph.node[i]['source']=='1':
                        sources.append(i)
                except: pass
            
            if len(sources)==0:
                print('Sources need to be set!')
                raise ValueError
                
        # get reduced graph and branches
        self.UpdateReducedDiGraph()
        DirectedBranches=self.UpdateDirectedBranchesFrom(self.ReducedDiGraph)
        DictDirectedBranches=self.UpdateDictDirectedBranchesFrom(self.DiGraph, DirectedBranches)
        colors=self.GenerateColors(self.ReducedDiGraph.number_of_edges())

        self.digraph_DictDirectedBranches=DictDirectedBranches

        for idx_source, source in enumerate(sources):
            
            nodes_levels=[]
            visited=[]
            
            for i in range(len(colors)):
                
                level=i+1
                branches=list(nx.bfs_edges(self.ReducedDiGraph, 
                                          source,
                                          depth_limit=level))
                
                nodes=[DictDirectedBranches[i] for i in branches]
                nodes=[h for j in nodes for k in j for h in k]
                nodes_levels.append(list(set(visited).symmetric_difference(set(nodes))))
                visited.extend(list( set(visited).union(set(nodes))))
                
            for idx, i in enumerate(nodes_levels):
                val=idx+1
                for j in i:
                    self.DiGraph.node[j]['branch']=val
 
    def LabelDiGraphBranching2(self, sources=[]):

        '''
        This funtion labels the branching level on 'directed' graph nodes 
        '''
        
        from VascGraph.Tools.CalcTools import LabelGraphBranchesManySources
        
        try: self.DiGraph
        except:
            print('UpdateDiGraph')
            return 
        
        for i in self.DiGraph.GetNodes():
            del self.DiGraph.node[i]['branch']
        
        if len(sources)==0:
            sources=[]
            for i in self.DiGraph.GetNodes():
                try:
                    if self.DiGraph.node[i]['source']=='1':
                        sources.append(i)
                except: pass
            
            if len(sources)==0:
                print('Sources need to be set!')
                raise ValueError
                
        b=LabelGraphBranchesManySources(self.DiGraph, sources)

    def LabelTreeBranching(self, root=None):
        
        try: self.Tree
        except:
            print('UpdateDiGraph')
            return 

        if root is None:
            for i in self.Tree.GetNodes():
                try:
                    if self.Tree.node[i]['root']=='1':
                        root=i
                except: pass           
            if root is None:
                print('Root for the garph is not assigned!')
                raise ValueError
             
                
        # get reduced graph and branches
        self.UpdateReducedTree()
        DirectedBranches=self.UpdateDirectedBranchesFrom(self.ReducedTree)
        DictDirectedBranches=self.UpdateDictDirectedBranchesFrom(self.Tree, DirectedBranches)
        colors=self.GenerateColors(self.ReducedTree.number_of_edges())                
              
        self.tree_DictDirectedBranches=DictDirectedBranches
        
        nodes_levels=[]
        visited=[]
        
        for i in range(len(colors)):
            
            level=i+1
            branches=list(nx.bfs_edges(self.ReducedTree, 
                                      root,
                                      depth_limit=level))
            
            nodes=[DictDirectedBranches[i] for i in branches]
            nodes=[h for j in nodes for k in j for h in k]
            nodes_levels.append(list(set(visited).symmetric_difference(set(nodes))))
            visited.extend(list( set(visited).union(set(nodes))))
            
        for idx, i in enumerate(nodes_levels):
            val=idx+1
            for j in i:
                self.Tree.node[j]['branch']=val
   
        
if __name__=='__main__':
    pass
    
    
    
    