#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:38:56 2019

@author: rdamseh
"""
import networkx as nx


class WriteSWC:
    
    def __init__(self, path, name, treegraph):
                
        '''
        Input: treegraph of the VascGraph.GeomGraph.DiGraph class
        '''
        self.path=path
        self.name=name
        
        # check tree
        if nx.is_tree(treegraph): pass
        else: 
            print('Cannot wirte non- tree graph!')
            return
        
        ee=list(nx.bfs_edges(treegraph, 105))
        e1=[i[0] for i in edges]   
        e1=treegraph.GetNodes()
        pos=treegraph.GetNodesPos()
        radii=treegraph.GetRadii()

        e2=[]
        
        for i in e1:
            pred=list(treegraph.predecessors(i))
            
            if pred:
                e2.append(pred[0])
            else:
                e2.append(-1)
        
        
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
        
        
        
        
        
        
        