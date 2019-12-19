#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:06:46 2018

@author: rdamseh
"""
import pandas as pn
from graphContraction import *

filename='g.xlsx'      
xls=pn.read_excel(filename)

try:
    x1, x2 = xls['x1'], xls['x2']
    y1, y2 = xls['y1'], xls['y2']
    z1, z2 = xls['z1'], xls['z2']

except:

    x1, x2 = xls['V1 x'], xls['V2 x']
    y1, y2 = xls['V1 y'], xls['V2 y']
    z1, z2 = xls['V1 z'], xls['V2 z']

ps=[(i,j,k) for i,j,k in zip(x1,y1,z1)] # start node
pe=[(i,j,k) for i,j,k in zip(x2,y2,z2)] # end node

# all nodes with their id's and pos's
p=list(set(ps).union(set(pe)))
pid=dict()
pos=dict()
for idx, i in enumerate(p):
    pid[str(i)]=idx
    pos[idx]=i


# graph
nodes=range(len(p))      
e=[(pid[str(i)],pid[str(j)]) for i,j in zip(ps,pe)]    
edges=[i for i in e if i[0]!=i[1]]

g=nx.Graph() 
g.add_nodes_from(nodes)      
g.add_edges_from(e)

for i in g.nodes():
    g.node[i]['pos']=np.array(pos[i])


nx.write_pajek(g,'g.pajek')
g=readPAJEK('g.pajek')
visG(g)



