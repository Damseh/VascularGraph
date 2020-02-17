#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:52:01 2019

@author: rdamseh
"""


from scipy.stats import ks_2samp as kstest
import numpy as np
import os

class ValidateRadius:
    
    def __init__(self, g_path, gtrue_path, fix_rad=True, mode='mean'):
                
        self.g_path=g_path
        self.gtrue_path=gtrue_path
        self.mode=mode
        self.fix_rad=fix_rad
    
    def __init(self, d_thr=2.0): 
        
        from VascGraph.GraphIO import ReadPajek
        
        try:
            g=ReadPajek(self.g_path).GetOutput()
            if self.fix_rad:
                 self.g=self.fixRad(g)
            else:
                self.g=g
            self.gtrue=ReadPajek(self.gtrue_path).GetOutput()
        except:
            g=ReadPajek(os.getcwd()+'/'+self.g_path).GetOutput()
            self.g=self.fixRad(g)
            self.gtrue=ReadPajek(os.getcwd()+'/'+self.gtrue_path).GetOutput()
            self.g_path=os.getcwd()+'/'+self.g_path
            self.gtrue_path=os.getcwd()+'/'+self.gtrue_path   
            
        nodes_exp=np.array(self.g.GetNodesPos())
        nodes_real=np.array(self.gtrue.GetNodesPos())
        
        radius_exp=np.array(self.g.GetRadii())[:,None]
        radius_real=np.array(self.gtrue.GetRadii())[:,None]


        dist1=[]
        for idx, i in enumerate(nodes_real):
            dist1.append(np.sum((i-nodes_exp)**2, axis=1))
            
        #real nodes with the corresponding exp. ones   
        idx1=np.argmin(dist1, axis=1)        
        d1=np.array([i[idx1[j]]**.5 for j, i in enumerate(dist1)]) 
        radius_exp_m=radius_exp[idx1]
        self.rad_real=radius_real[d1<d_thr]
        self.rad_exp_m=radius_exp_m[d1<d_thr]
        
        
        
        dist2=[]
        for idx, i in enumerate(nodes_exp):
            dist2.append(np.sum((i-nodes_real)**2, axis=1))
        #exp nodes with the corresponding real. ones   
        idx2=np.argmin(dist2, axis=1)    
        d2=np.array([i[idx2[j]]**.5 for j, i in enumerate(dist2)])
        radius_real_m=radius_real[idx2]
        self.rad_exp=radius_exp[d2<d_thr]
        self.rad_real_m=radius_real_m[d2<d_thr]
        
  
    def fixRad(self,g):

        '''
        fix radius values across graph branches
        '''
        
        from VascGraph.GeomGraph import GraphObject
        
        obj=GraphObject(g)
        obj.InitGraph()
        obj.UpdateReducedGraph()
        obj.UpdateDictBranches()
        red=obj.GetDictBranches()
        
        b=[]
        for i in red.keys():
            for j in red[i]:
                b.append(j)
                
        for i in b:
            r=np.array([g.node[k]['r'] for k in i])
            if self.mode=='mean':
                r=np.mean(r)
            elif self.mode=='max':
                r=np.max(r)
            else:
                r=np.median(r)                
            
            for k in i:
                g.node[k]['r']=r 
                
        return g 

       
    def Normalize(self, x):
        return (x-x.min())/(x.max()-x.min())  
    
    def MSEerror(self, x,y):
        e=(x-y)**2
        e=np.mean(e, axis=0)
        return e[0]
    
    def MPEerror(self, y_pred, y_true):
         
        m=y_true.min()
        
        if m<1.0:
            y_pred = y_pred+(1-m) 
            y_true = y_true+(1-m)
        
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100       
    
    def KSerror(self, x, y):
        e=kstest(x,y)
        return e[0]
    
    def GetScores(self, d_thr=10.0):
        
        '''
        Compute errors between radius values in the expiremental graph the true one
        
        Output:
            - mse: minimum mean square error
            - mpe: mean percentage error 
            - kse: kolomogorov smirnove measure 
        '''
        self.__init(d_thr=d_thr) # threshold of maximum distance when matching the two graphs
           
        self.rad_exp=self.Normalize(self.rad_exp)
        self.rad_real_m=self.Normalize(self.rad_real_m)
    
        mse=self.MSEerror(self.rad_exp, self.rad_real_m)
        mpe=self.MPEerror(self.rad_exp, self.rad_real_m)
        kse=self.KSerror(self.rad_exp[:,0], self.rad_real_m[:,0])        
        
        return mse, mpe, kse        

