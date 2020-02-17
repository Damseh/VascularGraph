#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:24:34 2019

@author: rdamseh
"""

import scipy as sp
import scipy.io as sio
import numpy as np
import networkx as nx
from VascGraph.GeomGraph import DiGraph
from VascGraph.GraphIO import ReadPajek, WritePajek
from VascGraph.Tools.VisTools import *
from VascGraph.Tools.CalcTools import *
from InitFlowSimulation import GraphObject, InitFlowSimulation

if __name__=="__main__":


    g=ReadPajek('/home/rdamseh/GraphPaper2018V1'+'/flow/test_network.di.pajek').GetOutput()
    metadata=sio.loadmat('/home/rdamseh/GraphPaper2018V1'+'/flow/test_network.di.pajek.metadata.mat')
    


    g_object= GraphObject(g)
    g_object.InitGraph()
    g=g_object.GetGraph()


    # ----------- assigne resistances ---------- #
    print('Computes resistances ...')
    flow=InitFlowSimulation(g, metadata)
    flow.ComputeResistance()
    flow.PrepareGraph()
    g=flow.Graph
    # ------------------------------------------#
    
        
    # -------- find cycles -------#
    sources=[]
    for i in g.GetNodes():
        try:
            if int(g.node[i]['inflow'])==True:
                sources.append(i)                    
        except: pass
        
    sinks=[]
    for i in g.GetNodes():
        try:
            if int(g.node[i]['outflow'])==True:
                sinks.append(i)           
        except: pass
              
    
    print('Generate cycles ...')
    
    cycles_basis=nx.cycle_basis(g)
    
    all_paths=[]
    for i in sources:
        for j in sinks:
            all_paths.append(nx.all_simple_paths(g, i, j)) # add pathes between sources and sinks
    
    cycles=[]
    
    # di_edges from closed loopes
    for i in cycles_basis:
        di_edg=zip(i[:-1],i[1:])
        di_edg.append((i[-1],i[0]))
        cycles.append(di_edg)
        
    # di_edges from closed loopes
    for paths in all_paths:
        for i in paths:
            di_edg=zip(i[:-1],i[1:])
            cycles.append(di_edg)    
    
    di_edges=[i for j in cycles for i in j]    
            
    def TransferAttributes(DiG, G):
        attr=['pos', 'r', 'type', 'branch']
        for att in attr:
            try:
                for i in DiG.GetNodes():
                    DiG.node[i][att]=G.node[i][att]
            except: print('No '+ att +'assigned to graph nodes!')
        
        edg=G.GetEdges()[0]
        attr_edg=G[edg[0]][edg[1]].keys()
        
        for att in attr_edg:
            for i in DiG.GetEdges():
                DiG[i[0]][i[1]][att]=G[i[0]][i[1]][att]
            
        for i in DiG.GetEdges():
            try:
                DiG[i[0]][i[1]]['inflow']=G[i[0]][i[1]]['inflow'] 
            except: pass
            try:
                DiG[i[0]][i[1]]['outflow']=G[i[0]][i[1]]['outflow'] 
            except: pass
            try:
                DiG[i[0]][i[1]]['pressure']=G[i[0]][i[1]]['pressure']  
            except: pass
 
        for i in DiG.GetNodes():
            try:
                DiG.node[i]['inflow']=G.node[i]['inflow'] 
            except: pass
            try:
                DiG.node[i]['outflow']=G.node[i]['outflow'] 
            except: pass
            
 
    print('Build directed graph ...')
    di_g=DiGraph()
    di_g.add_nodes_from(g.GetNodes())
    di_g.add_edges_from(di_edges)
    TransferAttributes(di_g, g)
    # ---------------------------------------------#
 

   
    # ------ build A from cycles ---------#
    
    print('Build A matrix ...')
    
    # assigne currents contibuting to each edge
    currents_in_edges=dict()
    for ed in di_g.GetEdges():
        cyc=[]
        for idx, c in enumerate(cycles):          
            if ed in c:
                cyc.append(idx)
        currents_in_edges[ed]=cyc
    
    #assigne currents contibuting to edges in each cycle
    currents_in_cycles=[]
    for cyc in cycles:
        curr=[]
        for ed in cyc:
            curr.append(currents_in_edges[ed])
        currents_in_cycles.append(curr)
 
  
    n_currents=len(cycles)
    A=sp.sparse.lil_matrix((n_currents, n_currents))
    
 
    # update A matrix 
    for row_id in range(n_currents): # iterate over rows in A
           
        # values of this row
        curr=np.zeros(n_currents)
        def f(j, val):
            curr[j]+=val
            
        void=[f(j, float(di_g[ed[0]][ed[1]]['res'])) 
                for i, ed in zip(currents_in_cycles[row_id], cycles[row_id]) 
                for j in i]
        
        
        ind1=np.array([row_id]*n_currents)
        ind2=range(n_currents)
        A[ind1,ind2]=curr
      
    
    # Build B matrix
    
    inflow_edges=[]
    outflow_edges=[]
    inflow_pressure=[]
    outflow_pressure=[]
    
    for i in di_g.GetEdges():
        try:
            if int(di_g[i[0]][i[1]]['inflow'])==1:
                inflow_edges.append(i)
                inflow_pressure.append(float(di_g[i[0]][i[1]]['pressure']))
        except: pass
        
        try:
            if int(di_g[i[0]][i[1]]['outflow'])==1:
                outflow_edges.append(i)
                outflow_pressure.append(float(di_g[i[0]][i[1]]['pressure']))

        except: pass
    
    
    cycles_with_in_p=[currents_in_edges[i] for i in inflow_edges]
    cycles_with_out_p=[currents_in_edges[i] for i in outflow_edges]
    

    B=np.zeros(n_currents)
    
    # summing inflow pressures
    if len(inflow_pressure)>1:
        for cyc, p in zip(cycles_with_in_p, inflow_pressure):        
            for row_id in cyc:
                B[row_id]+=p
    else:
        for row_id in cycles_with_in_p:
            B[row_id]+=inflow_pressure[0]
            
    # substracting outflow pressures
    if len(outflow_pressure)>1:
        for cyc, p in zip(cycles_with_out_p, outflow_pressure):        
            for row_id in cyc:
                B[row_id]+=p
    else:
        for row_id in cycles_with_out_p:
            B[row_id]+=outflow_pressure[0]
    
    # -------------------------#
    
    
    
    # ---------- Solve -------# 
    flow=sp.sparse.linalg.lsqr(A, B, atol=1e-010, btol=1e-010)[0] 
    # -------------------------#
    
    
    # ----- updating flow/direction at each edge -----#
    flow_in_edges=dict()
    for i in currents_in_edges.keys():
        flow_in_edges[i]=np.sum([flow[j] for j in currents_in_edges[i]])
    
    # set flow at eache edge
    for i in di_g.GetEdges():
        if flow_in_edges[i]>=0:
            di_g[i[0]][i[1]]['flow']=str(flow_in_edges[i])
            pass
        
        else:
            di_g.ReverseEdge(i)
            di_g[i[1]][i[0]]['flow']=str(-flow_in_edges[i])
            
    # set flow at each node
    for i in di_g.GetNodes():
        n=di_g.GetSuccessors(i)
        if len(n)>=1:
            di_g.node[i]['flow']=sum([float(di_g[i][j]['flow']) for j in n])
        else:
            n=di_g.GetPredecessors(i)
            di_g.node[i]['flow']=sum([float(di_g[j][i]['flow']) for j in n])    
                
   
    # set pressure at each edge
    flows=[float(di_g[i[0]][i[1]]['flow']) for i in di_g.GetEdges()]
    reses=[float(di_g[i[0]][i[1]]['res']) for i in di_g.GetEdges()]
    pressures=np.array(flows)*np.array(reses)
    
    for i, p in zip(di_g.GetEdges(), pressures):
        di_g[i[0]][i[1]]['pressure']=str(p)

    
    # set pressure at each node
    for i in di_g.GetNodes():
        n=di_g.GetSuccessors(i)
        if len(n)>=1:
            di_g.node[i]['pressure']=sum([float(di_g[i][j]['pressure']) for j in n])
        else:
            n=di_g.GetPredecessors(i)
            di_g.node[i]['pressure']=sum([float(di_g[j][i]['pressure']) for j in n])        
    
   
    # set volumes at each node
    for i in di_g.GetNodes():
        n=di_g.GetSuccessors(i)
        if len(n)>=1:
            di_g.node[i]['vol']=np.mean([float(di_g[i][j]['vol']) for j in n])
        else:
            n=di_g.GetPredecessors(i)
            di_g.node[i]['vol']=np.mean([float(di_g[j][i]['vol']) for j in n])     
            

    # set velocities on edges
    areas=[float(di_g[i[0]][i[1]]['area']) for i in di_g.GetEdges()]
    velocities=np.array(flows)/np.array(areas)
    for i, v in zip(di_g.GetEdges(), velocities):
        di_g[i[0]][i[1]]['velocity']=str(v)

    
    # set velocity at each node
    for i in di_g.GetNodes():
        n=di_g.GetSuccessors(i)
        if len(n)>=1:
            di_g.node[i]['velocity']=np.mean([float(di_g[i][j]['velocity']) for j in n])
        else:
            n=di_g.GetPredecessors(i)
            di_g.node[i]['velocity']=np.mean([float(di_g[j][i]['velocity']) for j in n])   
            
    # write di_graph
    WritePajek('', 'test_network_flow.di.pajek', di_g)


    
    
    
    

