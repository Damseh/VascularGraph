#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:42:10 2020

@author: rdamseh
"""


from VascGraph.GeomGraph import DiGraph
from VascGraph.Tools.CalcTools import fixG
import numpy as np

#from util import CreateMappings, CreateCylinderMappings


class ReadCASX:
    '''
    reading CASX graphs
    '''
    def __init__(self, path):
        
        self.path=path
        
        
    def ReadLines(self):
        
        diam=[]
        flow=[]
        pos=[]
        edges=[]
        
        lines = open(self.path, 'r').readlines()


        
        def readline(stat, l):
            
            if stat=='read_diam': 
                diam.append(float(l))

            if stat=='read_flow': 
                flow.append(float(l))
                
            if stat=='read_pos': 
                p=l.split(' ')
                p=[float(i) for i in p if i!='' and i!='\n']
                pos.append(p)   
                
            if stat=='read_edge': 
                e=l.split(' ')
                e=[int(i, 16) for i in e if i!='' and i!='\n']
                edges.append(e)  
                
        stat='none'
        
        for l in lines:
            
            stat0=stat
            chck=l.split(' ')[0]
            
            if chck=='//diameter:':
                stat = 'read_diam';
            elif chck=='//point':
                stat = 'read_pos';
            elif chck=='//arc':
                stat = 'read_edge';
            elif chck=='//flow':
                stat = 'read_flow';
            elif chck=='//end':
                stat = 'none';

            if stat!=stat0:
                stat0=stat
            else:
                readline(stat, l)


        return diam, pos, edges, flow
    
    
    def BuildGraph(self):
        
        '''
        read nodes with their pos and diameter, read edges
        '''
        
        d, p, e, f = self.ReadLines()

        G=DiGraph()
        G.add_nodes_from(range(len(p)))
        
        for i, j in zip(G.GetNodes(), p):
            G.node[i]['pos']=j
            
        e=np.array(e)-1
        G.add_edges_from(e.tolist())
            
        for i, j in zip(e, d):
            
            try:
                if j>G.node[i[0]]['d']:
                    G.node[i[0]]['d']=j
            except:
                G.node[i[0]]['d']=j
                    
            try:
                if j>G.node[i[1]]['d']:
                    G.node[i[1]]['d']=j
            except:
                G.node[i[1]]['d']=j
            
        self.G=fixG(G)

    def smooth_graph(self, gg, area_threshold):
        
	    from VascGraph.Tools.CalcTools import CycleAreaAll as area

	    g=gg.copy().to_undirected()
	    nodes=np.array([i for i in g.GetNodes() if len(list(g.neighbors(i)))==2])
	    nbrs=g.GetNeighbors()
	    pos=np.array(g.GetNodesPos())
	    nbrs=[i for i in nbrs if len(i)==2]
	    pos_nbrs=np.array([np.array([g.node[i]['pos'] for i in j]) for j in nbrs])
	    pos0=pos[nodes]
	    
	    # area
	    pos3=[[pos0[i], pos_nbrs[i,0,:], pos_nbrs[i,1,:]] for i in range(len(pos_nbrs))]
	    pos3=np.array(pos3)
	    a=area(pos3)
	    
	    # select
	    ind=np.where(a<area_threshold)[0]
	    pos0=pos0[ind]
	    pos_nbrs=pos_nbrs[ind]
	    nodes=nodes[ind]
	    
	    # centroid
	    newp=(pos0+np.sum(pos_nbrs, axis=1))/3
	#    vec=med-pos0
	#    newp=pos0+vec
	    
	    for i, p in zip(nodes, newp):
		    gg.node[i]['pos']=p

	    return gg


    def GetOutput(self, smoothing=False, area_threshold=1000):
        
        
        self.BuildGraph()
        
        if smoothing:
            self.G=self.smooth_graph(self.G, area_threshold=area_threshold)
        try:
            return self.G
        except:
            'Cannot read!'
               
        
if __name__=='__main__':
    
    
    path='Mimi/Memo1_Lesage/S1.201.withGroups.casx'
    #path='Mimi/Memo1_Lesage/E1_withGroups.casx'

    savepath='Mimi/jan072020/'
    
    file=ReadCASX(path)
    g=file.GetOutput()
    
