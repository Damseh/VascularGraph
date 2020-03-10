#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:03:10 2019

@author: rdamseh
"""

from VascGraph.GraphLab import GraphPlot, StackPlot
from VascGraph.GraphIO import ReadPajek, ReadStackMat 
from VascGraph.Tools.CalcTools import *
from VascGraph.GeomGraph import GraphObject 
from VascGraph.GeomGraph import DiGraph
#from VascGraph.Skeletonize import GenerateGraph, ContractGraph, RefineGraph, RefineGraphRadius

from mayavi import mlab
from VascGraph.Tools.VisTools import visG

import networkx as nx

class GenerateDiGraph(GraphObject):

    def __init__(self, Graph=None):
        GraphObject.__init__(self, Graph)
    
    def TransferAttributes(self, DiG, G):
      
        attr=['pos', 'r', 'type', 'branch', 'source', 'sink', 'root']
        
        for att in attr:
            for i in DiG.GetNodes():
                try:
                    DiG.node[i][att]=G.node[i][att]
                except: pass
        
        edg=G.GetEdges()[0]
        attr_edg=G[edg[0]][edg[1]].keys()
        
        for att in attr_edg:
            try:
                for i in DiG.GetEdges():
                    DiG[i[0]][i[1]][att]=G[i[0]][i[1]][att]
            except: print('No edge attribute: \''+ att +'\' assigned to graph!')
            
       
        ############## if attributes are set for some but not all of graph compartments 
        # ------------ edges ---------#
            
        for i in DiG.GetEdges():
            try:
                DiG[i[0]][i[1]]['inflow']=G[i[0]][i[1]]['inflow'] 
            except: pass
        
            try:
                DiG[i[0]][i[1]]['outflow']=G[i[0]][i[1]]['outflow'] 
            except: pass
        
            try:
                DiG[i[0]][i[1]]['pressure']=G[i[0]][i[1]]['pressure']  
            except: pass
 
        # --------- nodes ------------#
        for i in DiG.GetNodes():
            try:
                DiG.node[i]['inflow']=G.node[i]['inflow'] 
            except: pass
        
            try:
                DiG.node[i]['outflow']=G.node[i]['outflow'] 
            except: pass
        
            try:
                DiG.node[i]['sink']=G.node[i]['sink'] 
            except: pass
        
            try:
                DiG.node[i]['source']=G.node[i]['source'] 
            except: pass        

  
    def UpdateReducedDiGraphFromGraph(self):
        
        '''
        This function generated directed graphs from undirected graph 
            Note: Sources and Sinks has to be initialized in the object of this class
        '''
        
        
        def flip(s):
            return [(i[1], i[0]) for i in s]        
        
        def get_traversal(source):
    
            # nodes traversal
            non_tree_branches=list(nx.edge_dfs(self.ReducedGraph, source)) # passes nodes
            
            # edges traversal
            tree_branches=list(nx.bfs_edges(self.ReducedGraph, source)) # passes edges
            
            # to complete cycles (while avoiding self sink)
            bi_tree_branches=set(tree_branches).union(set(flip(tree_branches)))
            missing_edges=set(non_tree_branches).difference(bi_tree_branches)
            missing_eges=list(missing_edges)
        
            [tree_branches.append(i) for i in missing_edges]
            
            return tree_branches
        
        DirectedBranchesAll=[]
        for i in self.Sources:
            DirectedBranchesAll.append(get_traversal(i))
         
        try:    
            for i in self.Sinks:
                e=get_traversal(i)
                e=flip(e)
                DirectedBranchesAll.append(e)    
        except: 'Cannot use sinks to generate Directed edges!'
            
        #----- fix for multiple sources/sinks ----# 
        nodes=self.ReducedGraph.GetNodes()
        visited_edges=set()
        composed_edges=zip(*DirectedBranchesAll)
        edges=[]
       
        for i in composed_edges:
            
            edges_to_add=[]
            [edges_to_add.append(k) for k in i if (k[1],k[0]) not in edges_to_add]
            edges_to_add=list(set(edges_to_add))
            
            # chech if edge not added
            not_added=[k not in visited_edges for k in edges_to_add]
            edges_to_add=[z for z, k in zip(edges_to_add, not_added) if k]       
            edges.extend(edges_to_add)
            
            #add edges' nodes to visted nodes
            visited_edges=visited_edges.union(set(edges_to_add))
            visited_edges=visited_edges.union(set(flip(edges_to_add)))    
        
        DirectedBranches=edges
        
       #------ ReducedDiGraph -----#
        self.ReducedDiGraph=DiGraph()   
        self.ReducedDiGraph.add_edges_from(DirectedBranches)
        self.TransferAttributes(self.ReducedDiGraph, self.Graph)    

    def UpdateDirectedBranches(self):
        self.DirectedBranches=self.ReducedDiGraph.GetEdges()

    def UpdateDictDirectedBranches(self):
        Graph=self.Graph
        nodes_branches=[list(nx.all_shortest_paths(Graph, 
                                   source=e[0], 
                                   target=e[1], 
                                   weight='weight')) for e in self.DirectedBranches]
        self.DictDirectedBranches=dict(zip(self.DirectedBranches, nodes_branches))     
        
    def UpdateDiGraphFromGraph(self, Sources=[], Sinks=[]):
         
        self.Sources=Sources
        self.Sinks=Sinks

        if len(self.Sources)==0:
            raise ValueError
            return

 
        #------- Init ------#
        self.InitGraph()
        self.UpdateReducedGraph()
        self.UpdateReducedDiGraphFromGraph()
        self.UpdateDirectedBranches()
        self.UpdateDictDirectedBranches()
        
        #----- DiGraph ------#        
        self.DiGraph=DiGraph()   
        nodes_branches=[self.DictDirectedBranches[i] for i in self.DictDirectedBranches.keys()]
        
        #----- Create DiGraph -----#
        edges=[self.branch_to_edges(j) for i in nodes_branches for j in i]
        edges=[i for j in edges for i in j]
        edges=list(set(edges)) # might be not needed
            
        # add edges
        self.DiGraph.add_edges_from(edges)
        
        # add attributes
        self.TransferAttributes(self.DiGraph, self.Graph)
        
        for i in self.Sources:
            self.DiGraph.node[i]['source']='1'
 
        for i in self.Sinks:
            self.DiGraph.node[i]['sink']='1'
      
        
        
    def UpdateDiGraphFromGraph2(self, Sources=[], Sinks=[]):
         
        '''
        This function generate directed graphs from undirected graph and label di-graph nodes with the branch level 
        
        Input: 
            Source: inflow nodes
            Sinks: outflow nodes
            
            Note1: if Source or Sinks are not enterd, they will be 
            extarcted from the source/sink attributes that are on the input graph (self.Graph)
            
            Note2: This funtion is better and fatser than 'self.UpdateDiGraphFromGraph', which even do not 
            add branch labels on the di-graph generated
        '''


        if len(Sources)==0:
            Sources=[]
            for i in self.DiGraph.GetNodes():
                try:
                    if self.DiGraph.node[i]['source']=='1' or self.DiGraph.node[i]['source']==1:
                        Sources.append(i)
                except: pass
            
        if len(Sources)==0:
            print('Sources need to be set!')
            raise ValueError


        self.Sources=Sources
        self.Sinks=Sinks
        
        roots=self.Sources
        self.InitGraph()
        
        for i in self.Graph.GetNodes():
            try:
                del(self.Graph.node[i]['branch'])
            except:
                pass
        
        def flip(ed):
            return [(i[1], i[0]) for i in ed]
        
        def get_directed(gg, root):
            
            '''
            get directed graph using first breadth search giving one source
            '''
            edges=list(nx.bfs_edges(gg, root)) # directed edges
            old_edges=gg.GetEdges()
            
            g=gg.copy()
            self.TransferAttributes(g, gg)
            g.remove_edges_from(old_edges)
            g=g.to_directed()
            g.add_edges_from(edges)
              
            keep_edges=list(set(old_edges).difference(set(edges)))
            keep_edges=list(set(keep_edges).difference(set(flip(edges))))
            g.add_edges_from(keep_edges)
            
            return g
        
        def propagate(g, n, b):

            '''
            propagate 1 step from one node forward
            '''
            
            cont=1
            g.node[n]['branch']=b
            stat=0
            
            while cont>0:
                
                try:
                    n=g.GetSuccessors(n)
                except:
                    n=g.GetSuccessors(n[0])
                
                if len(n)==1:
                    g.node[n[0]]['branch']=b
                    stat=1
                else:
                    cont=1
                    break
                
            if len(n)==0:
                return 0, stat 
            else:
        #        try:
        #            dumb=g.node[n[0]]['branch'] # already passed (loopy structure)
        #            return 0, stat 
        #        except:
                return n, stat 
            
        def propagate_all(g, roots):
            
            '''
            assign branching labeles (brancing level) to a directed graph
            '''
            
            nextn=roots
            branch=1
            while 1:
                
                nxtn=[]
                stat=0
                
                for i in nextn:
                    n, s = propagate(g, i, branch)
                    stat+=s
                    if not n==0:
                        nxtn.append(n)
                
                branch+=1
                nextn=[j for i in nxtn for j in i]
                
                if stat==0:break
            
            branches=[]    
            no_branches=[] 
            for i in g.GetNodes():
                try:
                    b=g.node[i]['branch']
                    branches.append(b)
                except:
                    no_branches.append(i)
                    pass
            bmax=np.max(branches)
            bmax=bmax+1
            for i in no_branches:
                g.node[i]['branch']=bmax
                
            for e in g.GetEdges():
                
                g[e[0]][e[1]]['branch']= g.node[e[1]]['branch']
                
            return g
        
        
        def Transform(gg, roots):
            
            '''
            generate directed graphs when multiple sources are defined
            '''
            
            # generate seperate di-graphs from each of sources
            graphs=[]
            for r in roots:
                g=get_directed(gg, root=r)
                g=propagate_all(g, roots=[r])  
                graphs.append(g)
            
            # egdes
            edges=[i.GetEdges() for i in graphs]  
            
            # baseline
            e0=edges[0] # edges from baseline di-graph
            g0=graphs[0].copy() # baseline di-graph
            self.TransferAttributes(g0, graphs[0])
            
            if len(roots)>1:
                
                for i, graph in zip(edges[1:], graphs[1:]):
                    
                    # ------- fix branch level with edges of same orientation between baseline and new graph ---#
                    ed=np.array(list(set(i).intersection(set(e0)))) # edges that have similar orientaion between baseline and new graph
                    b_baseline=np.array([graph[k[0]][k[1]]['branch'] for k in ed]) # branch label on edges from di-g with differetn source
                    b_new=np.array([g0[k[0]][k[1]]['branch'] for k in ed]) # branch label on edges from baseline di-graph                    
                    
                    b=np.array([b_baseline, b_new])
                    ind=np.argmin(b, axis=0) 
                    
                    new_b=b_baseline[ind==0] # new branch labels from new graph
                    new_e= ed[ind==0]# edges to bes assigned new branch labels
                    
                    
                    for ee, bb in zip(new_e, new_b):
                        g0[ee[0]][ee[1]]['branch']=bb
                        g0.node[ee[0]]['branch']=bb                    
                    
                    
                    #-------------- pick right edges( right orientations) and their branch labels between baseline and new graph -------#
                    ed=np.array(list(set(i).difference(set(e0)))) # edges with different orientation between the baseline and new di-graph
                    ed_flip=np.array(flip(ed)) # flip edges to match that of from baseline
                    
                    ed_b=np.array([graph[k[0]][k[1]]['branch'] for k in ed]) # branch label on edges from di-g with differetn source
                    ed_flip_b=np.array([g0[k[0]][k[1]]['branch'] for k in ed_flip]) # branch label on edges from baseline di-graph
                    
                    b=np.array([ed_b, ed_flip_b])
                    ind=np.argmin(b, axis=0) # look for smaller branch level
                      
                    new_b=ed_b[ind==0]    
                    new_e=ed[ind==0]
                    remove_e=ed_flip[ind==0]
                    
                    g0.remove_edges_from(remove_e) # reomve edges with wrong orientation from baseline
                    g0.add_edges_from(new_e) # add right edge orientation 
                    
                    for ee, bb in zip(new_e, new_b):
                        g0[ee[0]][ee[1]]['branch']=bb
                        g0.node[ee[0]]['branch']=bb
                
            return g0
        
        g=Transform(self.Graph, roots=roots)
        self.DiGraph=g

    def UpdateReducedDiGraphFrom(self, DiGraph, ret=False):

        ReducedDiGraph=DiGraph.copy()
        
        cont=1
        while cont!=0:
            cont=0
            for i in ReducedDiGraph.GetNodes():
                p=ReducedDiGraph.GetPredecessors(i)
                s=ReducedDiGraph.GetSuccessors(i)
                if (len(p)==1 and len(s)==1):
                    ReducedDiGraph.remove_node(i)
                    ReducedDiGraph.add_edge(p[0], s[0], weight=1)
                    cont=1
                    
        if ret==True:
            return ReducedDiGraph
        else:
            self.ReducedDiGraph = ReducedDiGraph

    def UpdateDirectedBranchesFrom(self, ReducedDiGraph):
        return ReducedDiGraph.GetEdges()

    def UpdateDictDirectedBranchesFrom(self, Graph, DirectedBranches):
        nodes_branches=[list(nx.all_shortest_paths(Graph, 
                                   source=e[0], 
                                   target=e[1], 
                                   weight='weight')) for e in DirectedBranches]
        return dict(zip(DirectedBranches, nodes_branches)) 

    def UpdateReducedTree(self):
        
        try: self.Tree
        except:
            print('UpdateTreeFromDiGraph')
            return
            
        self.ReducedTree=self.UpdateReducedDiGraphFrom(self.Tree, ret=True)

    def UpdateReducedDiGraph(self):
        
        try: self.DiGraph
        except:
            print('UpdateDiGraph')
            return
            
        self.ReducedDiGraph=self.UpdateReducedDiGraphFrom(self.DiGraph, ret=True)
                    
    def UpdateTreeFromDiGraph(self, root, forest=False):
        
        try: self.DiGraph
        except:
            print('Run UpdateDiGraph!')
            return
            
        if forest==False:
            
            self.Tree=nx.maximum_spanning_arborescence(self.DiGraph.copy())
            
            # set tree root
            self.TreeRoot=root
                    
            self.TransferAttributes(self.Tree, self.DiGraph)
            self.UpdateReducedTree()
        
        else:
            self.Tree=nx.maximum_branching(self.DiGraph.copy())
            
            # set tree root
            self.TreeRoot=None
            self.TransferAttributes(self.Tree, self.DiGraph)
            self.UpdateReducedTree()
        
        self.Tree.node[root]['root']='1'
        
    # ----- Setters -------#
    def SetDiGraph(self, DiGraph):
        self.DiGraph=DiGraph
    
    def SetTree(self, Tree):
        
        self.Tree=Tree
        
        chck=0
        for i in self.Tree.GetNodes():
            try:
                self.Tree.node[i]['root']
                chch+1
            except:
                pass
        
        if chck==0:
            for i in self.Tree.GetNodes():
                if len(self.Tree.GetPredecessors(i))==0:
                    print ('Tree root: '+str(i))
                    self.Tree.node[i]['root']='1'
                    
        if chck>1: 
            print('This input is not a tree graph!')
            self.Tree=None
                    
        
    #------ Getters -------#
    def GetDirectedBranches(self):
 
        try:
            return self.DirectedBranches
        except:
            print('Run UpdateDiGraph!')

    def GetDictDirectedBranches(self):
 
        try:
            return self.DictDirectedBranches
        except:
            print('Run UpdateDiGraph!')
                
    def GetDiGraph(self):
 
        try:
            return self.DiGraph
        except:
            print('Run UpdateDiGraph!')
            
    def GetReducedDiGraph(self):
        
        try:
            return self.ReducedDiGraph
        except:
            print('Run UpdateDiGraph!')   
  
    def GetReducedTree(self):
        
        try:
            return self.ReducedTree
        except:
            print('Run UpdateTreeFromDiGraph!')   
            
    def GetTree(self):
        
        try:
            return self.Tree
        except:
            print('Run UpdateTreeFromDiGraph!')   
        
if __name__=='__main__':

    pass












