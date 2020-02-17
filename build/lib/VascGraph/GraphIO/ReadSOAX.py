#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:30:10 2019

@author: rdamseh
"""



import networkx as nx
from VascGraph.GeomGraph import Graph
import numpy as np
from VascGraph.Tools.VisTools import visG

class ReadSOAX:
    
    def __init__(self, path):
        
        self.path=path
        
        try:
            with open(self.path, 'r') as f:
                lines=f.readlines()
        except:
            print('Cannot read file!')
                
        start=[idx+1 for idx,i in enumerate(lines) if i[0] =='#']
        end=[idx-1 for idx,i in enumerate(lines) if i[0] =='#']
        end=end[1:]
        
        for idx, i in enumerate(lines[end[-1]:]):
            if i[0]=='[':
                end.append(end[-1]+idx)    
                break
        
        pathes=[]
    
        for s, e in zip(start, end):
            a=np.array([[float(j) for j in i.split(' ') if j!=''] for i in lines[s:e]])
            pathes.append(a[:,(2,3,4)])
            
            
        
        g=Graph()
        
        for i in pathes:
            
            n=g.number_of_nodes()
            nodes=range(n, n+len(i))
            g.add_nodes_from(nodes)
            
            for idx, k in enumerate(nodes):
                g.node[k]['pos']=np.array(i[idx])
            
            e1=nodes[0:-1]
            e2=nodes[1:]       
            
            e=[[k1, k2] for k1,k2 in zip(e1,e2)]
            
            g.add_edges_from(e)
        
        self.graph=g 
        
    def Update(self): pass
    
    def GetOutput(self):
        return self.graph
    
    
    
if __name__=='__main__':
    
    
    path='/home/rdamseh/GraphPaper2018V1/soaxData/mra/mra002.txt'
    f=ReadSOAX(path)

        
    from mayavi import mlab
    mlab.figure()
    visG(f.GetOutput())
    
    
    
    
    
    
    