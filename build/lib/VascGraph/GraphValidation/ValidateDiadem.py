#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:29:42 2019

@author: rdamseh
"""


import sys

# add VascGraph package to python path
try:
    sys.path.append('/home/rdamseh/GraphPaper2018V1/')
except: pass

import os
import subprocess
from VascGraph.GraphIO import WriteSWC, ReadSWC



class ValidateDiadem:
   
    '''
    Get diadem score from true and exp SWC tree files
    
    '''
    
    def __init__(self, true_graphs=None, exp_graphs=None, D=1, X=1, 
                 R=1, Z=1, zpath=True,
                 m=False):
        
        
        if type(true_graphs)!=list:
            self.true_graphs=[true_graphs]
        else:
            self.true_graphs=true_graphs
        
        if type(exp_graphs)!=list:
            self.exp_graphs=[exp_graphs]
        else:
            self.exp_graphs=exp_graphs
            
        self.scores=[]
        
        self.directory=os.path.dirname(os.path.realpath(__file__))
        self.D=D
        self.X=X
        self.R=R
        self.Z=Z
        
        if zpath:
            self.zpath='true'
        else:
            self.zpath='false'
            
        if m:
            self.m='true'
        else:
            self.m='false'
            
    def __GetSingleScore(self, true_graph, exp_graph):
        
        
        WriteSWC(path=self.directory+'/', name='true.swc', tree_graph=true_graph, root=0) # wirte true tree
        WriteSWC(path=self.directory+'/', name='exp.swc', tree_graph=exp_graph, root=0) # wirte exp tree
        
        true=self.directory+'/true.swc'
        test=self.directory+'/exp.swc'
        
        command=['java', '-jar', self.directory+'/DiademMetric.jar', 
                      '-G', true, 
                      '-T', test, 
                      '-m', self.m, 
                      '-D', str(self.D),
                      '-x', str(self.X),
                      '-R', str(self.R),
                      '--z-threshold', str(self.Z), 
                      '--z-path', self.zpath,
                      '-w','1']
        try:
            self.scores.append(subprocess.check_output(command).split()[1])
        except:
            self.scores.append(subprocess.check_output(command))

            
    def GetScores(self):

        for i, j in zip(self.true_graphs, self.exp_graphs):
            
            self.__GetSingleScore(i, j)
        
        return self.scores



if __name__=="__main__":
    
    
    
    
    path='/home/rdamseh/GraphPaper2018V1/validation/mra/trees/'
    truefile='truetree2.swc' 
    testfile='mytree2.swc'
    
    
    true_graph=ReadSWC(path+truefile).GetOutput()
    exp_graph=ReadSWC(path+testfile).GetOutput()
    
    
    diadem=ValidateDiadem([true_graph, true_graph], [exp_graph, exp_graph])
    s=diadem.GetScores()
    
    
    
    