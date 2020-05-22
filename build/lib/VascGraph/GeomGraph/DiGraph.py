#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:51:26 2019

@author: rdamseh
"""


from networkx.classes.digraph import DiGraph as DiG
import networkx as nx
from copy import deepcopy
import numpy as np

class DiGraph(DiG):
    
    def __init__(self, NodesPos=None, Edges=None, Radii=None, data=None, Types=None):        
        DiG.__init__(self, data)

        # attributes to be stored                 
        self.SetGeomGraph(NodesPos, Edges, Radii, Types)  
        self.Area=0           
        self.info=dict()
   
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
            
    def GetLabels(self, show=True):
        try:
            return [self.node[i]['label'] for i in self.nodes().keys()]   
        except:
                if show:
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
        
        '''
        find bifurcation nodes 
        if  bifurcation [i0, i1, ...], then find nodes that have i0 or i1 or ... bifurcations  
        if bifurcation=[] then find all of nodes including extremities
        '''
        
        nodes=[]
        
        for i in self.GetNodes():
        
            pred = self.GetPredecessors(i)
            succ = self.GetSuccessors(i)
            
            lpred=len(pred)
            lsucc=len(succ)
            
            l=lpred+lsucc
            
            if len(bifurcation)>0:
                
                if l in bifurcation:
                    nodes.append(i)
                elif lpred in bifurcation and lsucc==0:
                    nodes.append(i) 
                elif lsucc in bifurcation and lpred==0:
                    nodes.append(i)                    
            else:

                if lpred==1 and lsucc==1:
                    pass
                else:
                    nodes.append(i)
                    
        return nodes

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
                if g.node[i]['source']==1 or g.node[i]['source']=='1':
                    sources.append(i)
            except: pass
        
            try:
                if g.node[i]['sink']==1 or g.node[i]['sink']=='1':
                    sinks.append(i)
            except: pass   
        
        return sources, sinks    
    
    
    
    def PropagateTypes(self, cutoff=1, value=1, exclude_values=[2], other_value=3, backward=False):
            

            exclude_values.extend([value])
            
            try:
                self.node[self.GetNodes()[0]]['branch']
            except:
                print('Digraph should have \'branch\' labels! \n Ex. g.node[id][\'branch\']=1')
                return

            if backward:
                max_b=np.max([self.node[i]['branch'] for i in self.GetNodes()])
                branches=np.arange(max_b, max_b-cutoff, -1)
                print('Add types for branch levels:', branches)
                
            else:
                min_b=np.min([self.node[i]['branch'] for i in self.GetNodes()])
                branches=np.arange(min_b, min_b+cutoff, 1)
                print('Add types for branch levels:', branches)
            
            for b in branches:
                
                for i in self.GetNodes():
                    
                    if self.node[i]['branch']==b:
                        self.node[i]['type']=value
                        
                    if self.node[i]['type'] not in exclude_values:
                        self.node[i]['type']=other_value
            
    
    def FindSourcesSinks(self):
        '''
        label source and sink nodes using the keys 'source' and 'sink'
        '''
        sources=[]
        sinkes=[]
        for i in self.GetNodes():
            
            if len(self.GetPredecessors(i))==0 and len(self.GetSuccessors(i))>0:
                self.node[i]['source']=1
                
            if len(self.GetPredecessors(i))>0 and len(self.GetSuccessors(i))==0:
                self.node[i]['sink']=1                
        
    
    def LabelSegments(self): 
        
        gtest=self.copy().to_undirected()
        gtest.LabelSegments()
        
        for i in gtest.GetNodes():
            self.node[i]['label']=gtest.node[i]['label']
 
        del(gtest)
    
    def GetPathesDict(self):
    
        pathes=self.copy().to_undirected().GetPathesDict()
        
        # sort nodes in each path based direction
        def prop(s, t, nodes):

            p=[]
            p.append(s)

            nxt=s
            while 1:
                nxt=self.GetSuccessors(nxt)
                nxt=[k for k in nxt if (k in nodes)&(k!=s)&(k!=t)]

                try:
                    nxt=nxt[0]
                except:
                    p.append(t)
                    break # empty
                
                if nxt==t:
                    p.append(t)
                    break
                else:
                    p.append(nxt)
                    
            return p
        
        new_pathes=[]
        for item in pathes:
            try:
                n1, n2 = item[0]
            except:
                print(item)
                raise EOFError
                
            path_nodes = item[1]
            path1 = prop(n1, n2, path_nodes)
            new_key1=(n1, n2)
            path2 = prop(n2, n1, path_nodes)
            new_key2=(n2, n1)
            
            if len(path1)>len(path2):
                new_pathes.append([new_key1, path1]) 
            else:
                new_pathes.append([new_key2, path2]) 
            
            
        return new_pathes
    
    def LabelSegments2(self):   
        
         # ------------ split segments -------------#
        try:
            sources, sinks = self.GetSourcesSinks()
        except:
            self.FindSourcesSinks() 
            sources, sinks = self.GetSourcesSinks()
            
        s=sources[0]
        jnodes=self.GetJuntionNodes()
        
        # add empty labeling
        for i in self.GetNodes():
            self.node[i]['label']=[]   
            
        def label(g, s, jnodes, lb=0):
            
            '''
            prpagate over a directed graph to provide distinct labels for its segments given
            
            Inputs:
                - s: a source node
                - jnodes: bufrication/junction nodes including extremity nodes 
                - lb: the start label index (default=0)
            '''
            
            prop=list(nx.edge_dfs(g, s))
            double_tracing1=0
            double_tracing2=0
            
            countdt=0
            
            for e in prop:
                
                n1=e[0]
                n2=e[1]
                
                # ----- condition -----#
                
                # nodes type
                skln1=0
                skln2=0
                
                if not n1 in jnodes: skln1=1
                if not n1 in jnodes: skln2=1
                
                # increase label index
                plus=0
                
                #-----------------------------------------------------------------
                # label if n1 is skeletal node  (it needs to have just one label)
                #-----------------------------------------------------------------
                if len(g.node[n1]['label'])==0 and skln1:
                    g.node[n1]['label'].append(lb)
                    double_tracing1=0
                    
                elif len(g.node[n1]['label'])>0 and skln1:
                    double_tracing1=1 # this is not the first pass on node n1
                    countdt+=1
                    #print('double tracing!')
                #-----------------------------------------------------------------
    
                if len(g.node[n2]['label'])>0 and skln2: 
                    double_tracing2=1 # this is not the first pass on node n2   
                
                
                #-----------------------------------------------------------------
                # label if n1, n2 are bifurcation nodes (they accept to have multiple labels)
                #-----------------------------------------------------------------
                if n1 in jnodes: 
                    if not double_tracing2:
                        g.node[n1]['label'].append(lb)
                    
                if n2 in jnodes: 
                    if not double_tracing1:
                        g.node[n2]['label'].append(lb)  
                        plus=1 # increase label index just in this case
    
                if plus:
                    lb+=1
                
            print('Number of double tracing (source id %s): %s' %(s, countdt))
            return lb
        
        # iterate over sources and add labels
        lb=0
        for s in sources:
            lb=label(self, s, jnodes, lb)            
    
    
    def GetPathesDict2(self):
        
        jnodes = self.GetJuntionNodes()
        
        def get_pathes(g, source, jnodes):
            
            pathes=[]    
            nxts=g.GetSuccessors(source)
            
            if len(nxts)>0:# if this is not a sink

                for nxt in nxts:
                    
                    path=[]
                    path.append(source)
                    curr=nxt
                    
                    while 1:

                        if curr in jnodes:
                            path.append(curr)
                            break
                        else:
                            path.append(curr)
                        
                        curr=g.GetSuccessors(curr)
                        
                        if len(curr)>1:
                            if err==1:
                                print ('--Error when generating pathes!')
                                err=0
                            
                        curr=curr[0]
    
                    pathes.append(path)
                     
            return pathes 
                        
        pathes=[]                
        for node in jnodes:
            pathes.extend(get_pathes(self, source=node, jnodes=jnodes))
            
        PathesDict = dict()
        for i in pathes:
            PathesDict[(i[0], i[-1])]=i 
            
        return PathesDict
    
    
    def RefineExtremities(self):
        
        '''
        refine drected graphs by removing extremety nodes 
        that are not sinks
        '''
        
        n=self.number_of_nodes()
        
        sources, sinks = self.GetSourcesSinks()
        
        avoid=sources
        avoid.extend(sinks)
        
        while 1:
            rem=[]
            cont=0
            for i in self.GetNodes():
                if (len(self.GetSuccessors(i))+len(self.GetPredecessors(i)))==1 and i not in avoid:
                    rem.append(i)
                    cont+=1
                    
                if (len(self.GetSuccessors(i))+len(self.GetPredecessors(i)))==0:
                    rem.append(i)
                    cont+=1    
                
            self.remove_nodes_from(rem)
            if cont==0:
                break
                
        print('--None source/sink extremities refined! Number of nodes removed = %s'  %(n-self.number_of_nodes()))  
        
        
        
    def AddEdge(self, n1, n2, attr):
        
        '''
        attr: dictionary of attributes
        '''
        self.add_edge(n1, n2)
        for k in attr.keys():
            self[n1][n2][k]=attr[k]
            
            
    def RefineSegments(self, max_n=None, degree=15, fix_curve=True):
        
        
        refined_number=''
        refined_curve=''
        
        if degree>15:
            degree=15
            print('--\'degree\' should be less than 15')
        
        if type(max_n)==int:
            if max_n>55:
                max_n=55
                print('--\'max_n\' should be less than 15')
                
        def updatepos(n, pos):
            self.node[n]['pos']=pos
            
        def getind(l):
            step=int(l/degree)
            ind=list(range(0, l, step))
            return tuple(ind)        
        
        # fix if the number of nodes at a branch exceed the limit
        if max_n is not None and type(max_n)==int:
            
            pathes=self.GetPathesDict()
            pathes0=[k for k in pathes if len(k[1])>max_n]
            remnodes=[]
            
            for end, p in pathes0:
                    
                midp=p[1:-1] # avoid end nodes
                lenmidp=float(len(midp))
                ind=np.arange(0, lenmidp, lenmidp/(max_n-2)).astype(int)
                newp=[midp[k] for k in ind]
                
                if len(newp)>max_n-2:
                    print('yes')
                        
                #nods to be removed
                rem=list(set(midp).difference(set(newp)))
                remnodes.append(rem)
                
                #edges to be removed
                ed=list(zip(p[:-1], p[1:]))
                ed.extend([(p[0],p[1])])
                self.remove_edges_from(ed)
                
                # new edges
                newed=list(zip(newp[:-1], newp[1:]))
                newed.extend([(p[0],newp[0]), (newp[-1],p[-1])])
                self.add_edges_from(newed)

                
            remnodes=[k2 for k1 in remnodes for k2 in k1]
            self.remove_nodes_from(remnodes)
            refined_number='  Max number of nodes is '+str(max_n)+'.'
            
            
        # interpolating segment with bezier curves (for smoothing)
        if fix_curve:
            try:
                import bezier as bz
            except:
                print('To run this function, \'bezier\' sould be installed.')
                print('pip install bezier==0.9')    
    
        if fix_curve:
            
            print('--Refining graph segments ...')
            
            pathes=self.GetPathesDict()
            nodes=self.GetNodes()
            pos=self.GetNodesPos()
            posdict=dict(zip(nodes, pos))            
            
            for ends, p in pathes:
                
                nn=len(p)
                if nn>degree:
                    print(nn, degree)
                    poss = np.array([posdict[k] for k in p])
                    if len(poss)>=degree:
                        ind=getind(len(poss)) # get a maximim of 15 control points
                        poss=poss[ind, :]
                    x, y, z = poss[:,0], poss[:,1], poss[:,2]
                    xyz = np.asfortranarray([x,y,z])
                    print(xyz)
                    curve = bz.Curve(xyz, degree=degree)
                    stp=1/float(nn)
                    steps=np.arange(stp, 1-stp, stp)
                    new_pos=[np.ravel(curve.evaluate(i)) for i in steps]
                    dumb=[updatepos(n, pp) for n, pp in zip(p[1:-1], new_pos)]
                
            refined_curve='\n  Interpolating with bezier curvers of degree '+str(degree)+'.'

        if refined_curve!='' or refined_number!='':
            print(refined_number+refined_curve)
           
    def RefineRadiusOnSegments(self, rad_mode='max'):
        
        gtest=self.copy().to_undirected()
        gtest.RefineRadiusOnSegments(rad_mode=rad_mode) 

        for i in self.GetNodes():
            self.node[i]['r']=gtest.node[i]['r']
                    
