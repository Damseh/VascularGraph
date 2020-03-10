#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 21:10:46 2019

@author: rdamseh
"""

from networkx.classes.graph import Graph as G
import networkx as nx
from copy import deepcopy
import numpy as np

class Graph(G):
    
    def __init__(self, NodesPos=None, Edges=None, Radii=None, data=None, Types=None):        
        G.__init__(self, data=data)

        # attributes to be stored                 
        self.SetGeomGraph(NodesPos, Edges, Radii, Types)  
        self.Area=0    
        self.info=dict()
        
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
            
    def GetLabels(self):
        try:
            return [self.node[i]['label'] for i in self.nodes().keys()]   
        except:
                print('No branch labels assigned to graph nodes!')         
                return None   
            
            
    def GetAreas(self):
        try:
            return [self.node[i]['area'] for i in self.nodes().keys()]   
        except:
                print('No areas assigned to graph nodes!')         
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
    
    def GetJuntionNodes(self, bifurcation=[]):
        
        # nodes=set()
        
        # for i in bifurcation:
        #     u={node for node in self.GetNodes() if len(self.GetNeighbors(node))==i}
        #     nodes=nodes.union(u) 
            
        # return list(nodes)
        
        
        '''
        find bifurcation nodes 
        if  bifurcation [i0, i1, ...], then find nodes that have i0 or i1 or ... bifurcations  
        if bifurcation=[] then find all of nodes including extremities
        '''
        
        nodes=[]
        
        for i in self.GetNodes():
        
            nn = self.GetNeighbors(i)
            l=len(nn)
            
            if len(bifurcation)>0:
                if l in bifurcation:
                    nodes.append(i)
            else:
                if l!=2:
                    nodes.append(i)
        return nodes

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
            
    def LabelSegments(self):
        
        '''
        This funtion gives different id's for undirected graph segments
        
        '''
        
        def setlabel(x, l):
            self.node[x]['label']=[l]        
        
        # create a graph 'gtest' without junction nodes
        gtest=self.copy() 
        jnodes=gtest.GetJuntionNodes()
        gtest.remove_nodes_from(jnodes)
        subgraphs=list(nx.connected_component_subgraphs(gtest))
        
        # obtain sub graphs (they will be the branches) from the 'gtest'  and label their nodes
        label=-1
        for i in subgraphs:
            label+=1
            nodes=list(i.nodes())
            dumb=[setlabel(n, label) for n in nodes]
        
        # label juntion nodes 
        jed_added=[]
        
        for i in jnodes:
            
            # neighbors are not junc. nodes
            nbrs=self.GetNeighbors(i)
            j=[x for x in nbrs if x not in jnodes]
            labels=[self.node[k]['label'] for k in j]
            labels=[k2 for k1 in labels for k2 in k1]
            labels=np.unique(labels).tolist()
            self.node[i]['label']=labels
            
            
        # if some neighbors are junc. nodes (resulting when two jnodes connected directly in g)
        edd=[]
        for ed in self.GetEdges():
            if ed[0] in jnodes and ed[1] in jnodes:
                edd.append(ed)  
                
        for i in edd:
            label+=1
            try:
                self.node[i[0]]['label'].append(label)
            except:
                self.node[i[0]]['label'] = [label]                    
            try:
                self.node[i[1]]['label'].append(label)
            except:
                self.node[i[1]]['label'] = [label]                
                
        del(gtest)
            
            
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
                if g.node[i]['source']==1 or g.node[i]['source']=='1':
                    sources.append(i)
            except: pass
        
            try:
                if g.node[i]['sink']==1 or g.node[i]['sink']=='1':
                    sinks.append(i)
            except: pass           

        
        return sources, sinks        
    
    
        
        
    def ToDirected(self, Sources=[], Sinks=[]):
         
        '''
        This function generate directed graphs from undirected graph and label di-graph nodes with the branch level 
        
        Input: 
            Source: inflow nodes
            Sinks: outflow nodes
            
            Note1: if Source or Sinks are not enterd, they will be automaically
            extarcted from the source/sink attributes that are on the graph (Graph)
            
            Note2: This funtion is better and fatser than 'self.UpdateDiGraphFromGraph', which even do not 
            add branch labels on the di-graph generated
        '''
    
        import  VascGraph.Tools.CalcTools as calc
        from VascGraph.GeomGraph import GraphObject
    
    
    
        if len(Sources)==0:
            Sources, Sinks = self.GetSourcesSinks()
    
        if len(Sources)==0:
            print('Sources need to be set on graph!')
            raise ValueError
    
        roots = Sources
        g_object = GraphObject(self) 
        g_object.InitGraph()
        self = g_object.GetGraph()
        
        for i in self.GetNodes():
            try:
                del(self.node[i]['branch'])
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
            calc.TransferAttributes(g, gg)
            
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
                
                g[e[0]][e[1]]['branch'] = g.node[e[1]]['branch']
                
            return g
        
        
        def Transform(gg, roots):
            
            '''
            generate directed graphs with single or multiple sources being defined
            '''
            
            # generate seperate di-graphs with their branch labels from each of sources
            graphs=[]
            for r in roots:
                g=get_directed(gg, root=r)
                g=propagate_all(g, roots=[r])  
                graphs.append(g)
            
            # egdes
            edges=[i.GetEdges() for i in graphs]  
            
            # baseline
            e0 = edges[0] # edges from baseline di-graph
            g0 = graphs[0].copy() # baseline di-graph
            
            calc.TransferAttributes(g0, graphs[0])
            
            if len(roots)>1:
                
                for i, graph in zip(edges[1:], graphs[1:]):
                    
                    # ------- fix branch level with edges of same orientation between baseline and new graph ---#
                    ed = np.array(list(set(i).intersection(set(e0)))) # edges that have similar orientaion between baseline and new graph
                    b_update = np.array([graph[k[0]][k[1]]['branch'] for k in ed]) # branch label on edges from di-g with different source
                    b_original = np.array([g0[k[0]][k[1]]['branch'] for k in ed]) # branch label on edges from baseline di-graph                    
                    
                    b = np.array([b_update, b_original])
                    
                    
                    ind = np.argmin(b, axis=0) 
                    new_b = b_update[ind==0] # new branch labels from new graph
                    new_e = ed[ind==0]# edges to bes assigned new branch labels
                    
                    
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
          
        return Transform(self, roots=roots)
      
    
    def RefineExtremities(self):
        
        '''
        refine drected graphs by removing extremety nodes 
        that are not sinks
        '''
        
        n=self.number_of_nodes()
        
        sources, sinks = self.GetSourcesSinks()
        
        ss=sources
        ss.extend(sinks)
        
        while 1:
            rem=[]
            cont=0
            for i in self.GetNodes():
                if len(self.GetNeighbors(i))==1 and i not in ss:
                    rem.append(i)
                    cont+=1
               
                if len(self.GetNeighbors(i))==0:
                    rem.append(i)
                    cont+=1                    
                    
            self.remove_nodes_from(rem)
            if cont==0:
                break
            
        
        print('--None source/sink extremities refined! Number of nodes removed = %s'  %(n-self.number_of_nodes()))  
        
        
    def GetPathesDict(self):
    
        try:
            labels=self.GetLabels(show=False)
            dumb=labels[0]
        except:
            self.LabelSegments()
            labels=self.GetLabels()
        
        maxlabel=np.max([j for i in labels for j in i])
        pathes=dict(zip(range(maxlabel+1), [[] for i in range(maxlabel+1)]))
    
        for i in self.GetNodes():
            l=self.node[i]['label']
            dumb=[pathes[j].append(i) for j in l]  
            
        # save pathes with their end nodes
        jnodes=self.GetJuntionNodes()
        pathes_ends=[]
        for i in pathes.keys():
            path=pathes[i]
            ends=[]
            for j in path:
                if j in jnodes:
                    ends.append(j) 
            pathes_ends.append(tuple(ends))    
            
        pathes=zip(pathes_ends  , [pathes[i] for i in pathes.keys()])
            
        return pathes        
        
    def RefineRadiusOnSegments(self, rad_mode='max'):
           
        def updaterad(n, r):
            self.node[n]['r']=r
            
        # interpolating segment with bezier curves (for smoothing)
        pathes=self.GetPathesDict()
        jnodes=self.GetJuntionNodes()
        for ends, p in pathes:
            if len(p)>2:
                rad=[self.node[k]['r'] for k in p]
                
                if rad_mode=='max':
                    r=np.max(rad)
                if rad_mode=='mean':
                    r=np.mean(rad)                    
                if rad_mode=='median':
                    r=np.median(rad) 
                    
                dumb=[updaterad(n, r) for n in p] 

        for node in jnodes:
            nbrs=self.GetNeighbors(node)
            rad=[self.node[k]['r'] for k in nbrs]
            r=np.mean(rad)     
            self.node[node]['r']=r
            
        print('--Radii on segments are refined by taking the '+rad_mode+'.')
            
        
            
    def AddEdge(self, n1, n2, attr):
        
        '''
        attr: dictionary of attributes
        '''
        self.add_edge(n1, n2)
        for k in attr.keys():
            self[n1][n2][k]=attr[k]   
    
    
    
    
if __name__=='__main__':
    pass
