#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:48:36 2019

@author: rdamseh
"""

import numpy as np
import networkx as nx

from VascGraph.Tools.CalcTools import *

from VascGraph.GeomGraph import Graph

import scipy as sp


# get neighbors d index
def neighbors(shape):
    dim = len(shape)
    block = np.ones([3]*dim)
    block[tuple([1]*dim)] = 0
    idx = np.where(block>0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx-[1]*dim)
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx, acc[::-1])

def mark(img): # mark the array use (0, 1, 2)
    nbs = neighbors(img.shape)
    img = img.ravel()
    for p in range(len(img)):
        if img[p]==0:continue
        s = 0
        for dp in nbs:
            if img[p+dp]!=0:s+=1
        if s==2:img[p]=1
        else:img[p]=2

def idx2rc(idx, acc):
    rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i,j] = idx[i]//acc[j]
            idx[i] -= rst[i,j]*acc[j]
    rst -= 1
    return rst
    
def fill(img, p, num, nbs, acc, buf):
   
    back = img[p]
    img[p] = num
    buf[0] = p
    cur = 0; s = 1;
    
    while True:
        p = buf[cur]
        for dp in nbs:
            cp = p+dp
            if img[cp]==back:
                img[cp] = num
                buf[s] = cp
                s+=1
        cur += 1
        if cur==s:break
    return idx2rc(buf[:s], acc)

def trace(img, p, nbs, acc, buf):
    
    c1 = 0; c2 = 0;
    newp = 0
    cur = 0

    while True:
        buf[cur] = p
        img[p] = 0
        cur += 1
        for dp in nbs:
            cp = p + dp
            if img[cp] >= 10:
                if c1==0:c1=img[cp]
                else: c2 = img[cp]
            if img[cp] == 1:
                newp = cp
        p = newp
        if c2!=0:break
    return (c1-10, c2-10, idx2rc(buf[:cur], acc))
   
def parse_struc(img):
    nbs = neighbors(img.shape)
    acc = np.cumprod((1,)+img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    pts = np.array(np.where(img==2))[0]
    buf = np.zeros(131072, dtype=np.int64)
    num = 10
    nodes = []
    for p in pts:
        if img[p] == 2:
            nds = fill(img, p, num, nbs, acc, buf)
            num += 1
            nodes.append(nds)

    edges = []
    for p in pts:
        for dp in nbs:
            if img[p+dp]==1:
                edge = trace(img, p+dp, nbs, acc, buf)
                edges.append(edge)
    return nodes, edges
    
# use nodes and edges build a networkx graph
def build_graph(nodes, edges, multi=False):
    
    graph = nx.MultiGraph() if multi else Graph()
    
    for i in range(len(nodes)):
        graph.add_node(i, pts=nodes[i], pos=nodes[i].mean(axis=0))
        
    for s, e, pts in edges:
        l = np.linalg.norm(pts[1:]-pts[:-1], axis=1).sum()
        graph.add_edge(s, e, pts=pts, weight=l)
    return graph

def buffer(ske):
    buf = np.zeros(tuple(np.array(ske.shape)+2), dtype=np.uint16)
    buf[tuple([slice(1,-1)]*buf.ndim)] = ske
    return buf

def build_sknw(ske, multi=False):
    buf = buffer(ske)
    mark(buf)
    nodes, edges = parse_struc(buf)
    return build_graph(nodes, edges, multi)
    
# draw the graph
def draw_graph(img, graph, cn=255, ce=128):
    acc = np.cumprod((1,)+img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    for idx in graph.GetNodes():
        pts = graph.node[idx]['pts']
        img[np.dot(pts, acc)] = cn
    for (s, e) in graph.GetEdges():
        eds = graph[s][e]
        for i in eds:
            pts = eds[i]['pts']
            img[np.dot(pts, acc)] = ce



class Skel3D:
    
  
    def __init__(self, image, method=1):
        
        try:
            from skimage.morphology import skeletonize_3d as skel
        except:
            print('To run this function, \'scikit-image\' sould be installed.')
            return  
        
        self.image=image
        self.Graph=None
        self.method=method
        
        
    def Update(self, ConnectionParam=4, Resolution=0.75):
        
        ske = skel(self.image)
        self.ske = ske.astype(np.uint16)
        
        if self.method==2:
            self.Graph = build_sknw(ske)
        
        elif self.method==1: 
            self.ConnectionParam = ConnectionParam
            self.Resolution = Resolution    
            self.__Read()
            self.__ReadNodes()
            self.__CreateConnections()
            self.__CreateGraph()            


    def __Read(self):
        
        pos=np.where(self.ske>0)
        self.X=pos[1]
        self.Y=pos[2]
        self.Z=pos[0]


    def __ReadNodes(self):
        
        # graph nodes from centerline
        self.GraphNodes = np.array([self.X, self.Y, self.Z]).T
        self.GraphNodes = self.GraphNodes.astype('float')
        self.GraphNodes, ClustersPos, Clusters= AssignToClusters(self.GraphNodes, resolution=self.Resolution)
        
 #       self.GraphRadius=[np.max([self.__Radius[i] for i in j]) for j in Clusters]
        self.NNodes=len(self.GraphNodes)                         
            
    
    def __CreateConnections(self):      
        
        # connections from graph nodes
        self.Connections=[]
        
        length=len(self.GraphNodes)
        Tree=sp.spatial.cKDTree(self.GraphNodes)
        c=[Tree.query(i, k=self.ConnectionParam)[1] for i in self.GraphNodes]
        c=np.array(c)
         
        connections=[]
        
        for i in range(self.ConnectionParam):
            
            # obtain and fix connection from tree.query
            if i>0:
                cc=c[:,(0,i)]
                exclude=np.where(cc[:,1]>=len(self.GraphNodes))  
                cc[exclude]=0
                connections.append(cc)
            
        self.Connections=np.vstack(tuple(connections))
                    


    def __CreateGraph(self):
        
        # build graph  
        
        self.Graph=Graph()
            
        ind=np.array(range(self.NNodes))
        self.Graph.add_nodes_from(ind)  
            
        for i, p in zip(ind, self.GraphNodes):
            self.Graph.node[i]['pos']=p
#            self.Graph.node[i]['r']=r
            
        self.Graph.add_edges_from(self.Connections)      
        self.Graph.remove_edges_from(self.Graph.selfloop_edges())
        self.Graph=fixG(self.Graph)
     


    def GetOutput(self):
        if self.Graph is not None:
            return self.Graph








if __name__ == '__main__':
    
    pass
