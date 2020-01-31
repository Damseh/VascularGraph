#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:07:58 2019

@author: rdamseh
"""

import sys
try:
    sys.path.append('/home/rdamseh/GraphPaper2018V1')
except: pass
from VascGraph.GeomGraph import Graph, DiGraph
from VascGraph.Tools.CalcTools import fixG
import numpy as np
import networkx as nx


class ReadPajek():
    
    def __init__(self, filename=None, mode=''):
        '''
        Input: 
            mode='' undirected graph read;
            mode='di' read as direcetd graph
        '''
        self.filename=filename
        self.G=None
        self.mode=mode
    """
    Reads a Pajek file storing a Graph 
    following the format used in this mudule.
    
    Input:
        "filename": The full directory to Graph file.
    """
    
    
    def ReadFile(self):
        
        G_init=nx.read_pajek(self.filename) 
        
        self.G_init=G_init
        
        if self.mode=='di':
            G=DiGraph()
        else:
            G=Graph() 
        
        # build geometry
        for i in list(G_init.nodes()):
            
            node=G_init.node[i]      
            
            # add node
            #n=int(node['id'].encode())
            
            n=int(i)
            G.add_node(n)
            
            #add position
            
            if sys.version_info[0]>=3:
                pos=node['pos']
            else:
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
                        try:
                            xyz.append(float(value[1]))
                        except:
                            value=value[1].split(',')
                            xyz.append(float(value[0]))
                    except:
                        try:
                            value=pos[j].split(']')
                            try:
                                xyz.append(float(value[0]))
                            except:
                                value=value[1].split(',')
                                xyz.append(float(value[0]))                                
                        except: 
                            try:
                                value=pos[j].split(',')
                                xyz.append(float(value[0]))
                            except: pass                  
                    
                    
            G.node[n]['pos']=np.array(xyz)
            
            # add label
            try:
                yORn=node['node'].encode()
                if yORn=='False':
                    G.node[n]['node']=False
                else:
                    G.node[n]['node']=True
            except:
                pass
               
            # add radius
            try:
                radius=node['d'].encode()
                G.node[n]['d']=float(radius)
            except:
                pass
            
            # add radius
            try:
                radius=node['r'].encode()
                G.node[n]['r']=float(radius)
            except:
                pass
            
            # add radius
            try:
                radius=node['r'].encode()
                G.node[n]['r']=float(radius.split('[')[1].split(']')[0])
            except:
                pass
            
            
            # add radius
            try:
                radius=node['d'].encode()
                G.node[n]['d']=float(radius.split('[')[1].split(']')[0])
            except:
                pass

                       
            # add type
            try:
                t=node['type'].encode()
                G.node[n]['type']=int(t)
            except:
                pass
 
            # add branch
            try:
                b=node['branch'].encode()
                G.node[n]['branch']=int(b)
            except:
                pass
            
            # add inflow
            try:
                b=node['inflow'].encode()
                G.node[n]['inflow']=str(int(b))
            except:
                pass
            
            # add outflow
            try:
                b=node['outflow'].encode()
                G.node[n]['outflow']=str(int(b))
            except:
                pass
            
            
            # add sink
            try:
                b=node['sink'].encode()
                G.node[n]['sink']=str(int(b))
            except:
                pass
            
            # add source
            try:
                b=node['source'].encode()
                G.node[n]['source']=str(int(b))
            except:
                pass    
            
            # add root
            try:
                b=node['root'].encode()
                G.node[n]['root']=str(int(b))
            except:
                pass   
            
            # add flow
            try:
                b=node['flow'].encode()
                G.node[n]['flow']=float(b)
            except:
                pass     

            # add pressure
            try:
                b=node['pressure'].encode()
                G.node[n]['pressure']=float(b)
            except:
                pass    

            # add velocity
            try:
                b=node['velocity'].encode()
                G.node[n]['velocity']=float(b)
            except:
                pass  
            
            # add velocity
            try:
                b=node['so2'].encode()
                G.node[n]['so2']=float(b)
            except:
                pass  

            # add velocity
            try:
                b=node['po2'].encode()
                G.node[n]['po2']=float(b)
            except:
                pass              
                                        
        #build Topology
        raw_edges=list(G_init.edges()) 
        edges=[(int(i[0]),int(i[1])) for i in raw_edges]
        G.add_edges_from(edges)
        
        for i, j in zip(raw_edges, edges):
            try:
                G[j[0]][j[1]]['res']=G_init[i[0]][i[1]][0]['res'] 
            except: pass
            try:
                G[j[0]][j[1]]['flow']=G_init[i[0]][i[1]][0]['flow'] 
            except: pass
            try:
                G[j[0]][j[1]]['pressure']=G_init[i[0]][i[1]][0]['res'] 
            except: pass
        
            try:
                G[j[0]][j[1]]['inflow']=G_init[i[0]][i[1]][0]['inflow'] 
            except: pass

            try:
                G[j[0]][j[1]]['outflow']=G_init[i[0]][i[1]][0]['outflow'] 
            except: pass
        
            try:
                G[j[0]][j[1]]['branch']=G_init[i[0]][i[1]][0]['branch'] 
            except: pass
        
        self.G=G
        
    def GetOutput(self):
        
        self.ReadFile()
                
        if self.G is not None:
            return fixG(self.G)
        
        
        
        
if __name__=='__main__':

    file_name='/home/rdamseh/GraphPaper2018V1/VascGraph/test_network_reduced.pajek'                    
    g=ReadPajek(filename=file_name)
    graph=g.GetOutput()   
        
        
        
        
        
        
        
        
        
        
        
        
        
        