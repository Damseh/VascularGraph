#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:38:56 2019

@author: rdamseh
"""
import networkx as nx
from VascGraph.GeomGraph import DiGraph

class WriteSWC:
    
    def __init__(self, path, name, tree_graph, root):
                
        '''
        Input: treegraph of the VascGraph.GeomGraph.DiGraph class
        '''
        self.path=path
        self.name=name
        
        # check tree
        if nx.is_tree(tree_graph): pass
        else: 
            print('Cannot wirte non- tree graph!')
            return
        
        
        def fix_indexing(g_old, root):
            
            bfs=list(nx.bfs_predecessors(treegraph, root))
            old_indices=[root]
            old_indices.extend([i[0] for i in bfs])
            
            new_indices=range(len(old_indices))
            mapping={old_indices[i]:new_indices[i] for i in new_indices}
           
            g=DiGraph()
            g.add_nodes_from(new_indices)
            
            for i in old_indices:
                g.node[mapping[i]]['pos']=g_old.node[i]['pos']
                g.node[mapping[i]]['r']=g_old.node[i]['r']
            
            edges_old=g_old.GetEdges()
            edges_new=[[mapping[e[0]], mapping[e[1]]] for e in edges_old]
            
            g.add_edges_from(edges_new)

            return g
            
        # fix indixing to start with 0 at root node
        treegraph=tree_graph.copy()        
        treegraph=fix_indexing(treegraph, root)
        
        # get info
        pos=treegraph.GetNodesPos()
        radii=treegraph.GetRadii()
       
        bfs=dict(nx.bfs_predecessors(treegraph, 0))        
        
        e1=treegraph.GetNodes()
        e2=[]
        
        for i in e1:
            if i==0:
                e2.append(-1)
            else:
                e2.append(bfs[i])                   

        self.__write_graph(e1, e2, pos, radii)
        
        
    def __write_graph(self, e1, e2, pos, radii):
        
        if self.name.split('.')[-1]!='swc':
            self.name=self.name+'.swc'
            
        with open(self.path+self.name, 'w') as out:
            lines=[str(n)+
                   ' '+
                   str(2)+
                   ' '+
                   str(p[0])+
                   ' '+
                   str(p[1])+
                   ' '+
                   str(p[2])+
                   ' '+
                   str(r)+
                   ' '+
                   str(pred)+
                   '\n' for n, pred, p, r in zip(e1, e2, pos, radii)]
            out.writelines(lines)
            
    def Update(self):pass
    


if __name__=='__main__':
    pass        
        
        
        
        
        
        
        