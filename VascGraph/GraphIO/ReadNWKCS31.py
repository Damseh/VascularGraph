#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:55:17 2020

@author: rdamseh
"""

import numpy as np
from VascGraph.GraphLab import GraphPlot
from VascGraph.GeomGraph import Graph, DiGraph
from VascGraph.Tools.CalcTools import fixG

class ReadNWKCS31:
    '''
    reading graphs and their attributes from path_net (.NWK) and path_attr (.CS31)
    '''
    def __init__(self, path_net, mode='di', path_attr=None):
        
        '''
        Input:
            mode: if 'di' -> output graph will be directed, otherwise will be undirected
        '''
        self.path_net=path_net 
        self.path_attr=path_attr
        self.stat='noread'
        self.mode=mode        
        
    def ReadNet(self):
        '''
        read from 'path_net'
        '''
        def readpos(l):
            p=l.split()
            return p   
                
        def readedge(l):
            e=l.split()[1:3]
            e=[int(i, 16) for i in e]
            return e 
                    
        def updatestat(l):
            if l.find('Dimension')!=-1:
                self.stat='readpos'
            if l.find('Faces')!=-1:
                self.stat='readedge'
                
        def checkread(l):
            if l.find('(')!=-1: return 0
            elif l.find(')')!=-1: return 0
            elif len(l.split())==0: return 0
            else: return 1
                
        pos=[]
        edges=[]
        
        lines = open(self.path_net, 'r').readlines()
            
        for l in lines:
            updatestat(l)
            if checkread(l):
                if self.stat=='readpos':
                    pos.append(readpos(l))
                elif self.stat=='readedge':
                    edges.append(readedge(l))
                                   
        pos=np.array(pos).astype(float)
        edges=np.array(edges).astype(int)

        return pos, edges
            
    def ReadAttr(self):
        '''
        read from path_attr
        '''
        
        attr=dict()
        
        def read(l):
            p=l.split()
            return p   
     
        def checkstat(l):
            if l.find('vector')!=-1 or l.find('Vector')!=-1: 
                try:
                    att=l.split('(')[1].split('=')[1]
                except:
                    return 0
                chk1=att.find('|')
                if chk1!=-1:
                    att=att[0:chk1]
                else:
                    att=att.split()[0]
                return att
            else: return 0
            
        def checkread(l):
            if l.find('(')!=-1: return 0
            elif l.find(')')!=-1: return 0
            elif len(l.split())==0: return 0
            else: return 1            
        
        lines = open(self.path_attr, 'r').readlines()

        att=0
        for l in lines:
            chk=checkstat(l)
            if chk!=0:
                att=chk
                print('--Reading att: '+att+' ...')
            if checkread(l) and att!=0:
                try:
                    attr[att].append(read(l))
                except:
                    attr[att]=[]
                    attr[att].append(read(l))

        return attr   
    
    
    
    def BuildGraph(self):
        
        '''
        1 Build graph based on info from 'path_net'
        2 Set attributes to graph nodes based on info from 'path_attr'
        '''

        def setatt(g,e,v,name):
            for i, j in zip(e, v):
                try:
                    if j>G.node[i[0]][name]:
                        G.node[i[0]][name]=j
                except:
                    G.node[i[0]][name]=j
                try:
                    if j>G.node[i[1]][name]:
                        G.node[i[1]][name]=j
                except:
                    G.node[i[1]][name]=j            
            return g
        
        p, e = self.ReadNet()
        
        if self.path_attr is not None:
            self.attr = self.ReadAttr()

        if self.mode=='di':
            G=DiGraph()
        else:
            G=Graph() 
            
        G.add_nodes_from(range(len(p)))
        
        for i, j in zip(G.GetNodes(), p):
            G.node[i]['pos']=j
            
        e=np.array(e)-1
        G.add_edges_from(e.tolist())
            
        # set diameter/radius
        try:
            d=np.array(self.attr['Dia']).ravel().astype(float)
            G=setatt(G,e,d,'d')
            G=setatt(G,e,d/2,'r')
        except:
            print('--Cannot set diam!')
            
        # set flow
        try:
            flow=np.array(self.attr['flow']).ravel().astype(float)
            G=setatt(G,e,flow,'flow')
        except:
            print('--Cannot set flow!')            
            
        # set po2
        try:
            po2=np.array(self.attr['ppO2']).ravel().astype(float)
            G=setatt(G,e,po2,'po2')
        except:
            print('--Cannot set po2!')             

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
    

    net_filename='au.vGrant.Oxygen200x200x200model1.nwk'
    attr_filename='au.vGrant.Oxygen200x200x200model1.cs31'
    

    file=ReadNWKCS31(path_net=net_filename, 
                 path_attr=attr_filename)
    g=file.GetOutput()


    '''
    lines = open(attr_filename, 'r').readlines()
    chk_key='Dimension'
    def chk(i,j):
        return i==j
        
    check=[chk(i.split(chk_key)[0], i) for i in lines]
    check_result = np.where(np.array(check)==0)[0]
    if len(check_result)==0:
        print('Item not found!!!')
        
    lines_=[]
    ind=[]
    for idx, l in enumerate(lines):
        if l.find('(')!=-1:
            lines_.append(l)
            ind.append(idx)
    attr=file.attr
    '''
    # read/write
    from VascGraph.GraphIO import WritePajek, ReadPajek
    namepajek=''.join(attr_filename.split('.')[:-1])+'.pajek'
    WritePajek('',namepajek, g)
    g=ReadPajek(namepajek, mode='di').GetOutput()
 
    from OxyGraph import OxyGraph
    og=OxyGraph(g)
    og.AddType()
    og.AddVelocityFromFlow()
    og.AddSo2FromPo2()
    og.BuildDirectionsGraph()
    g=og.g
    
    gplot = GraphPlot(new_engine=True)
    gplot.Update(g)
    gplot.SetTubeRadius(3)
    gplot.SetTubeRadiusByScale(True)
    gplot.SetTubeRadiusByColor(True)
  
