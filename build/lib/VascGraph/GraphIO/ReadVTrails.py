#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:56:28 2019

@author: rdamseh
"""


import os
import sys

# add VascGraph package to python path
try:
    sys.path.append(os.getcwd())
except: pass


import numpy as np
from scipy import sparse


class ReadVTrails:
    
    def __init__(self, filepath, sampling=1):
        
        from VascGraph.GeomGraph import Graph
        
        try:
            import h5py
        except:
            print('To run this function, \'h5py\' sould be installed.')
            return   

        # ---- read ----#
        f=h5py.File(filepath, 'r')
        refs=[i[0] for i in f.get('GeodesicMSTs/CGPathContinuous')]
        pathes=[np.array(f[i]).T for i in refs]
        
        data=f.get('GeodesicMSTsMatrix/M/data')
        ir=np.array(f.get('GeodesicMSTsMatrix/M/ir'))
        jc=np.array(f.get('GeodesicMSTsMatrix/M/jc'))




        # ----- sampling of pathes nodes ----#
        ind=[len(i)/sampling for i in pathes]
        ind=[np.array(range(i))*sampling for i in ind]
        pathes=[i[indx] for i, indx in zip(pathes, ind)]
        
        
        # ----- build graph from pathes -----#
        path_ext=[]
        g=Graph()
        for path in pathes:
            n=g.number_of_nodes()
            nodes=np.array(range(len(path)))+n
            e1=nodes[1:]
            e2=nodes[:-1]
            e=np.array([e1,e2]).T
            
            path_ext.append([nodes[0], nodes[-1]])
            
            g.add_nodes_from(nodes)
            g.add_edges_from(e)
            
            for node, pos in zip(nodes, path):
                g.node[node]['pos']=np.array([pos[1], pos[0], pos[2]])
        
 
        # ------- connection between pathes ----#
        path_ext=np.array(path_ext)
        a = sparse.csc_matrix((data, ir, jc))
        ind1, ind2 = np.where(a.todense()>0)       
        
        e=[]
        for i,j in zip(ind1,ind2): 
            
            ee=[[path_ext[i][0], path_ext[j][1]],
             [path_ext[i][1], path_ext[j][0]],
             [path_ext[i][0], path_ext[j][0]],
             [path_ext[i][1], path_ext[j][1]]]
            
            
            poss=np.array([[g.node[k[0]]['pos'], g.node[k[1]]['pos']] for k in ee])
            poss= poss[:,0,:]-poss[:,1,:]
            norm=np.linalg.norm(poss, axis=1)
            indx=np.where(norm==norm.min())[0][0]
            e.append(ee[indx])
        
        g.add_edges_from(e)
             
        self.graph=g 
    
        
    def Update(self): pass
    
    def GetOutput(self):
        return self.graph  