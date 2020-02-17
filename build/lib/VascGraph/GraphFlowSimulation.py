#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:47:19 2019

@author: rdamseh
"""

import numpy as np
from scipy.optimize import curve_fit
from VascGraph.Tools.CalcTools import *

class GraphObject:
    
    def __init__(self, Graph=None):
        self.Graph=Graph
            
    def InitGraph(self):
        
        '''
        Obtain a gaph containing only bifurcation nodes
        '''
        if self.Graph is None:
            
            print('Run SetGraph!')
            return
        else:
            pass
        
        self.JunctionNodes=list(self.GetJuntionNodes())  
        
        # assign  weights
        for v1, v2 in self.Graph.GetEdges():
            if (v1 in self.JunctionNodes or v2 in self.JunctionNodes):
                self.Graph[v1][v2]['weight']=1
            else:
                self.Graph[v1][v2]['weight']=0       
        
    def GetJuntionNodes(self):
                    
        nodes=[node for node in self.Graph.GetNodes() if 
           len(self.Graph.GetNeighbors(node))!=2]
    
        return nodes

    def GetNotJuntionNodes(self):
                    
        return set(self.Graph.GetNodes()).difference(self.GetJuntionNodes())

    def branch_to_edges(self, p):
        p1=p[:-1]
        p2=p[1:]
        return [(i,j) for i,j in zip(p1,p2)]
        
        
    def UpdateReducedGraph(self):
        
        self.ReducedGraph=self.Graph.copy()

        cont=1
        while cont!=0:
            cont=0
            for i in self.ReducedGraph.GetNodes():
                k=self.ReducedGraph.GetNeighbors(i)
                if len(k)==2:
                    self.ReducedGraph.remove_node(i)
                    if (k[0] in self.JunctionNodes or k[1] in self.JunctionNodes):
                        self.ReducedGraph.add_edge(k[0], k[1], weight=1)
                    else:
                        self.ReducedGraph.add_edge(k[0], k[1], weight=0)
                    cont=1
                        
    def GetGraph(self):
        return self.Graph
    
    def SetGraph(self, Graph):
        self.Graph=Graph
        
    def GetReducedGraph(self):
        
        try:
            return self.ReducedGraph
        except: return

class polyfit3D:
    
    def __init__(self, x, y, f, order=5):
        
        self.order=order
        X=np.array([x,y])
        self.a1, self.a2, self.a3, \
        self.a4, self.a5, self.b1, self.b2, \
        self.b3, self.b4, self.b5, self.c=curve_fit(self.func, X, f)[0]
        
    def func(self, X, a1, a2, a3, 
             a4, a5, b1, b2, b3, b4, b5, c):
        
        on=np.array([0,0,0,0,0])
        on[:self.order]=1
        
        z=on[4]*a5*X[0]**5 + on[3]*a4*X[0]**4 + on[2]*a3*X[0]**3 + on[1]*a2*X[0]**2+ on[0]*a1*X[0] +\
          on[4]*b5*X[1]**5 + on[3]*b4*X[1]**4 + on[2]*b3*X[1]**3 + on[1]*b2*X[1]**2+ on[0]*b1*X[1] + c
        return z            
 
    def compute(self, x, y):
        
        return self.func([x, y], self.a1, self.a2, self.a3,
                         self.a4, self.a5, self.b1, self.b2,
                         self.b3, self.b4, self.b5, self.c)
                   
class Viscosity(polyfit3D):
           
    def __init__(self, order=3):
        
        typee=[10, 10, 10, 10, 10, 10 ,30 ,20 ,20, 20, 20, 20, 20]
        diam=[30.5, 24.4, 19.5, 15.6, 12.5, 10, 8, 12, 15, 18.7, 23.4, 29.3, 36.6]
        viscosity=[2.49, 2.34, 2.25, 2.2, 2.16, 2.12, 2.1, 2.15, 2.18, 2.22, 2.32, 2.51, 2.7] 
        polyfit3D.__init__(self, typee, diam, viscosity, order=order)

        
class Hematocrit(polyfit3D):
           
    def __init__(self, order=4):
        
        typee=[10, 10, 10, 10, 10, 10 ,30 ,20 ,20, 20, 20, 20, 20]
        diam=[30.5, 24.4, 19.5, 15.6, 12.5, 10, 8, 12, 15, 18.7, 23.4, 29.3, 36.6]
        hematocrit=[14.1, 11.5, 9.8, 9.4, 9.1, 8.8, 8.5, 8.8, 8.8, 8.8, 11.1, 15.1, 18] 
        polyfit3D.__init__(self, typee, diam, hematocrit, order=order)
        
 
def test_fitting():
    
    '''
    test the fitting for viscosity and hematocrit
    '''
    v=Viscosity()       
    h=Hematocrit()       

    diam=[30.5, 24.4, 19.5, 15.6, 12.5, 10, 8, 12, 15, 18.7, 23.4, 29.3, 36.6]
    viscosity=[2.49, 2.34, 2.25, 2.2, 2.16, 2.12, 2.1, 2.15, 2.18, 2.22, 2.32, 2.51, 2.7] 
    hematocrit=[14.1, 11.5, 9.8, 9.4, 9.1, 8.8, 8.5, 8.8, 8.8, 8.8, 11.1, 15.1, 18] 

    x=np.arange(min(diam), max(diam), .01)
    y1=v.compute(10, x)
    y2=h.compute(10, x)
    
    plt.figure()
    plt.scatter(diam[6:], viscosity[6:])
    plt.scatter(x, y1)

    plt.figure()
    plt.scatter(diam[6:], hematocrit[6:])
    plt.scatter(x, y2)
    
        
class GraphFlowSimulation:
    
    def __init__(self, graph=None, metadata=None):
        
        self.Graph=graph
        self.metadata=metadata
        self.arteriol_pressure=60.0
        self.venule_pressure=25.0

    def ComputeResistance(self):
            
        print('Computes resistances ...')

        edges=np.array(self.Graph.GetEdges())
        nodes1=edges[:,0]
        nodes2=edges[:,1]
        
        pos1=np.array([self.Graph.node[i]['pos'] for i in nodes1])
        pos2=np.array([self.Graph.node[i]['pos'] for i in nodes2])
        length=np.sqrt(np.sum((pos2-pos1)**2, axis=1))
        
        radius=np.array([(self.Graph.node[i]['r']+\
                          self.Graph.node[j]['r'])/2.0 for i, j in zip(nodes1, nodes2)])
        area=(radius**2)*np.pi
        volume=area*length
        diameter=radius*2.0
        
        typee=np.array([max( self.Graph.node[i]['type'], 
                            self.Graph.node[j]['type']) for i, j in zip(nodes1, nodes2)])
        v=Viscosity()
        viscosity=v.compute(typee, diameter)
        
        # -------------------------------------------------#
        resistance=128*viscosity*length/(np.pi*diameter**4)
        #--------------------------------------------------#
        
        for i, j, res, v, a in zip(nodes1, nodes2, resistance, volume, area):
            self.Graph[i][j]['res']=str(res)
            self.Graph[i][j]['area']=str(a)
            self.Graph[i][j]['vol']=str(v)
        
    def GetCloseGraph(self): 
        
        # ------ closed graph with no end nodes ------#
        
        print('Prepare graph ...')
        # sources and sinks
        sources=self.metadata['sources'][0]
        sinks=self.metadata['sinks'][0]
        
        # end nodes only
        sources_sinks=[i for i in sources]
        sources_sinks.extend([i for i in sinks])
        end_nodes=[i for i in self.Graph.GetNodes() if len(self.Graph.GetNeighbors(i))==1]
        
        for i in sources_sinks: end_nodes.remove(i) 
        

#        # in lfow and out flow  
#        #inflow_edges
        for i in sources:
            n=self.Graph.GetNeighbors(i)
            self.Graph.node[i]['inflow']='1'
            for j in n:
                self.Graph[i][j]['pressure']=str(self.arteriol_pressure)
                self.Graph[i][j]['inflow']='1'
#        
#       #outflow_edges
        for i in sinks:
            n=self.Graph.GetNeighbors(i)
            self.Graph.node[i]['outflow']='1'
            for j in n:
                self.Graph[i][j]['pressure']=str(-self.venule_pressure)
                self.Graph[i][j]['outflow']='1'
    
        self.Graph=fixG(self.Graph)
        # ----------------------------------------------#
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        