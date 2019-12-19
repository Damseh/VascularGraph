#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:29:31 2019

@author: rdamseh
"""

from VascGraph.Tools.CalcTools import fixG, FullyConnectedGraph 


import networkx as nx
import numpy as np
import scipy as sp

import scipy.io as sio 

def PostProcessMRIGraph(graph, upper_distance=7.0, k=5):
    
    '''
    This function reconnect seperated segments of MRI graph
    '''
    pos_all=np.array(graph.GetNodesPos()) # pos all nodes
    nodes_all=graph.GetNodes()
    
    # ----- overconnecting the graph --------#
    
    #subgraphs
    graphs=list(nx.connected_component_subgraphs(graph))
    
    # obtain end nodes/nodes and their positions in each segment
    nodes=[i.GetNodes() for i in graphs]
    end_nodes=[i.GetJuntionNodes(bifurcation=[0, 1]) for i in graphs] # end nodes in each subgraph
    end_nodes_pos=[np.array([graph.node[i]['pos'] for i in j]) for j in end_nodes] # pos of end nodes
        
    # obtain closest node from ther segments to an end node from current segment
    closest_nodes=[]
    for end_n, n, end_p in zip(end_nodes, nodes, end_nodes_pos): #iterate over each segment
        
        other_nodes=list(set(nodes_all).symmetric_difference(set(n)))
        other_pos=np.array([graph.node[i]['pos'] for i in other_nodes])
        
        
         # closest nodes in graph to current segment end nodes ...
         # except for nodes in current segment  
        mapping=dict(zip(range(len(other_nodes)), other_nodes))
        
        ind_notvalid=len(other_pos)
        tree=sp.spatial.cKDTree(other_pos)
        closest=[tree.query(i, k=k, distance_upper_bound=upper_distance)[1][1:] for i in end_p]
        closest=[[i for i in j if i!=ind_notvalid] for j in closest] # fix from query
        closest=[[mapping[i] for i in j] for j in closest] # fix indixing
        closest_nodes.append(closest)
    
    
    # create new graph amd add new edges
    graph_new=graph.copy()
    closest_nodes=[i for j in closest_nodes for i in j ]
    end_nodes=[i for j in end_nodes for i in j ]
    edges_new=[[i,k] for i, j in zip(end_nodes, closest_nodes) for k in j]
    graph_new.add_edges_from(edges_new)
    
    graphs_new=list(nx.connected_component_subgraphs(graph_new))
    print('Elements in each connected component: ')
    print([len(i) for i in graphs_new])
    
    # refine overconnectivity
    from VascGraph.Skeletonize import RefineGraph
    final_graph=FullyConnectedGraph(graph_new)
    refine=RefineGraph(final_graph)
    refine.Update(AreaParam=50.0, PolyParam=10)
    final_graph=fixG(refine.GetOutput())
    
    return final_graph




def RegisterGraph(target, source, mode='affine', tolerance=1e-3):
    
    # registration
    import pycpd as cpd
    from functools import partial
    nodes_before=np.array(source.GetNodesPos())
    if mode=='affine':
        new_pos=cpd.affine_registration(np.array(target.GetNodesPos()), nodes_before, tolerance=tolerance)
    else:
        new_pos=cpd.rigid_registration(np.array(target.GetNodesPos()), nodes_before, tolerance=tolerance)
        
    new_pos.register(callback=None)
    r=new_pos.updateTransform()
    
    nodes_after=new_pos.TY
    
    for idx, i in zip(source.GetNodes(), nodes_after):
        source.node[idx]['pos']=i
    
    return source




def RegCP(target, source, mode='affine'):
    
    def __init__(self, target=None, source=None, mode='affine'):
        
        if target is not None or source is not None:
            self.Update(target=target, source=source, mode=mode)
        
        
        
    def Update(self, target, source, mode='affine'):
        
        # registration
        import pycpd as cpd
        
        self.source=source
                
        if mode=='affine':
            self.reg=cpd.affine_registration(target, self.source, tolerance=1e-3)
        else:
            self.reg=cpd.rigid_registration(target, self.source, tolerance=1e-3)
            
        self.reg.register(callback=None)
        


class RegGraph:
    
    def __init__(self, target=None, source=None, mode='affine'):
        
        if target is not None or source is not None:
            self.Update(target=target, source=source, mode=mode)
        
        
        
    def Update(self, target, source, mode='affine'):
        
        # registration
        import pycpd as cpd
        from functools import partial
        
        self.source=source
        
        nodes_before=np.array(self.source.GetNodesPos())
        
        if mode=='affine':
            self.reg=cpd.affine_registration(np.array(target.GetNodesPos()), nodes_before, tolerance=1e-3)
        else:
            self.reg=cpd.rigid_registration(np.array(target.GetNodesPos()), nodes_before, tolerance=1e-3)
            
        self.reg.register(callback=None)
        self.reg.updateTransform()   
        
        
        
    def GetOutput(self):
        
        r=self.reg.updateTransform()
        
        nodes_after=self.reg.TY
        
        for idx, i in zip(self.source.GetNodes(), nodes_after):
            self.source.node[idx]['pos']=i
        
        return self.source


def ReadGraphfromMat(filename):
    
    f=sio.loadmat(filename)
    mat=f['im2'][0,0]
    
    nX=int(mat['nX'])
    nY=int(mat['nY'])
    nZ=int(mat['nZ'])
     
    scale=mat['Hvox'][0]

    xx=int(nX*scale[0])
    yy=int(nY*scale[1])
    zz=int(nZ*scale[2])
       
    # read nodes 
    pos=mat['nodePos'].astype(float)
    radii=mat['nodeDiam'].T
    
    # read edges
    edg=(mat['nodeEdges']).astype('int')
    connections=[]
    for i in range(len(edg)):
        connections.append((edg[i,0]-1,edg[i,1]-1))
    
    from VascGraph.GeomGraph import Graph
    G=Graph()
    G.add_nodes_from(range(pos.shape[0]))
    G.add_edges_from(connections)
    
    for i, p, r in zip(G.GetNodes(), pos, radii):
        G.node[i]['pos']=p
        G.node[i]['r']=r

    return G
