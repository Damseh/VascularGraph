#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 09:56:34 2019

@author: rdamseh
"""

from Code.Tools.CalcTools import *
from Code.GeomGraph import Graph

class ReadPAJEK():
    
    def __init__(self, filename):
        self.filename=filename
        self.G=None
        
    """
    Reads a Pajek file storing a Graph 
    following the format used in this mudule.
    
    Input:
        "filename": The full directory to Graph file.
    """
    
    
    def __ReadFile(self):
        
        G_init=nx.read_pajek(self.filename)    
        G=Graph() 
        
        # build geometry
        for i in range(G_init.number_of_nodes()):
            node=G_init.node[str(i)]      
            
            # add node
            n=int(node['id'].encode())
            G.add_node(n-1)
            
            #add position
            pos=node['pos'].encode()
            pos=pos.split(' ')
            
            xyz=[]
            for j in range(len(pos)):         
                try:                
                    value=float(pos[j])
                    xyz.append(value)
                except:
                    try:
                        value=pos[j].split('[')
                        xyz.append(float(value[1]))
                    except:
                        try:
                            value=pos[j].split(']')
                            xyz.append(float(value[0]))
                        except : pass
            G.node[i]['pos']=np.array(xyz)
            
            # add label
            try:
                yORn=node['node'].encode()
                if yORn=='False':
                    G.node[i]['node']=False
                else:
                    G.node[i]['node']=True
            except:
                pass
                         
        #build Topology
        connections_=G_init.edges()   
        connections=[[int((connections_[i][0]).encode()), int((connections_[i][1]).encode())] for i in range(len(connections_))]
        G.add_edges_from(connections)
        
        self.G=G
        
        
        def GetOuput(self):
            try:
                self.__ReadFile()
            except: 
                print('Cannot read pajek file!') 
            
            if G is not None:
                return self.G
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        