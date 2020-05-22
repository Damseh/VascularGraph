#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 08:56:18 2019

@author: rdamseh
"""


import os


from VascGraph.Tools.CalcTools import prunG, reduceG,\
getMiddleGraph, rescaleG, \
findNodes, getBranches, fixG, getCoreGraph



from VascGraph.Tools.VisTools import visG


import numpy as np

try:
    from mayavi import mlab
except: pass

try:
    from matplotlib import pyplot as plt
except: pass


class ValidateNetMets:
    
    def __init__(self, Gr, Ge,
                 rescale=False,
                 middle=False,
                 prune=False,
                 outputfolder='results',
                 sigma=[10,20,30,40,50,60]):


        self.Gr=Gr.copy()
        self.Ge=Ge.copy()
        
        if prune:           
            self.Gr=prunG(Gr.copy())
            self.Ge=prunG(Ge.copy())      
            
        #middle graphs
        if middle:
            self.Gr=getMiddleGraph(self.Gr, middle)
            self.Ge=getMiddleGraph(self.Ge, middle)
        
        
        #rescale graphs
        if rescale:
            self.Gr=rescaleG(self.Gr)
            self.Ge=rescaleG(self.Ge)
        
        #find graphs vertices
        self.points_real=np.array(self.Gr.GetNodesPos()) 
        self.points_exp=np.array(self.Ge.GetNodesPos())
               
        #find burifications (junction nodes)
        self.idNodes_real, self.nodes_real = findNodes(self.Gr)
        self.nodes_real=np.array(self.nodes_real)
        #
        self.idNodes_exp, self.nodes_exp = findNodes(self.Ge)
        self.nodes_exp=np.array(self.nodes_exp)

        # num of all nodes
        self.n_nodes_r=np.shape(self.nodes_real)[0]
        self.n_nodes_e=np.shape(self.nodes_exp)[0]

        #reduced graphs
        self.G_real_reduced=reduceG(self.Gr.copy())
        self.G_exp_reduced=reduceG(self.Ge.copy())

        # get branches
        self.branches1=getBranches(self.Gr)
        self.branches2=getBranches(self.Ge)

       
        self.outputfolder=outputfolder
        self.sigma=sigma
        


    def vis(self, save=False, name=None, cam=None):
       
        from VascGraph.Tools.VisTools import setCam, createCam
        from VascGraph.GraphLab import GraphPlot
        
        def plot(g, color):
            
            gplot=GraphPlot()
            gplot.Update(g)
            gplot.SetGylphSize(.01)
            gplot.SetTubeRadius(2)
            gplot.SetTubeColor(color)
            gplot.SetTubeRadiusByScale(True)
            
        
        bgcolor=(1,1,1)
        
        if cam is None:
            position = [1194.8393680906522, 1491.5272445674307, -874.4021568391549]
            focal_point = [257.15006008258143, 256.92547521800316, 330.6489784843938]
            view_angle = 30.0
            view_up = [-0.4853531757850406, -0.39346331460859185, -0.7807809646838195]
            clipping_range = [940.3721291401878, 3256.3268137240707]   
                
            cam=createCam(position=position,
                          focal_point=focal_point,
                          view_angle=view_angle,
                          view_up=view_up,
                          clipping_range=clipping_range)        
            


        
        # visulize matching
        mlab.figure(bgcolor=bgcolor)    
        plot(self.Gr, color=(.3,.3,.8))
        plot(self.Gcore_real, color=(.3,.3,.8))
        plot(self.Gcompared_real, color=(.9,.9,.1))
        
        setCam(cam)
        
        if save:
            mlab.savefig(name+'_FN.png', size=(1024,1024))
        #
        mlab.figure(bgcolor=bgcolor) 
        plot(self.Ge, color=(.3,.3,.8))
        plot(self.Gcore_exp, color=(.3,.3,.8))
        plot(self.Gcompared_exp, color=(.9,.9,.1))      
        setCam(cam)
        
        if save:
            mlab.savefig(name+'_FP.png', size=(1024,1024))
            
            
    def matchG(self):
                
        # REAL TO EXP
        self.dist1=[]
        for idx, i in enumerate(self.nodes_real):
            self.dist1.append(np.sum((i-self.nodes_exp)**2, axis=1))
        #real nodes with the corresponding exp. ones   
        self.idx1=np.argmin(self.dist1, axis=1)        
        self.d1=[i[self.idx1[j]]**.5 for j, i in enumerate(self.dist1)]
        self.idNodes_exp_m=np.array(self.idNodes_exp)[self.idx1]
        self.nodes_exp_m=self.nodes_exp[self.idx1]   
    
    
        # EXP TO REAL   
        self.dist2=[]
        for idx, i in enumerate(self.nodes_exp):
            self.dist2.append(np.sum((i-self.nodes_real)**2, axis=1))
        #exp nodes with the corresponding real. ones   
        self.idx2=np.argmin(self.dist2, axis=1)    
        self.d2=[i[self.idx2[j]]**.5 for j, i in enumerate(self.dist2)]       
        self.idNodes_real_m=np.array(self.idNodes_real)[self.idx2]
        self.nodes_real_m=self.nodes_real[self.idx2]   
  
    
    
    def scoresG(self, portion=[.99], 
                   save=False,
                   foldername=None):
        
        sigma=self.sigma
        
        self.matchG()
        
        if foldername:
            pass
        else:
            foldername=self.outputfolder
            
        def decideThresh(v, portion):
            vals,bins=np.histogram(v,bins=1000)
            vals=vals.astype(float)/sum(vals)
            s=0
            thresh=0
            for idx, i in enumerate(vals):
                s+=i
                if s>portion:
                    thresh=bins[idx]
                    break
            return thresh
        
        # match nodes and get G scores
        self.GFNR=[]     
        self.GFPR=[]
        
        for j in portion:
            
            thresh1=decideThresh(self.d1,j)
            thresh2=decideThresh(self.d2,j)
        
            g_FNR_=[]    
            for i in sigma:
                v1=np.array(self.d1)
                v1=v1*(v1<thresh1)
                v2=1-np.exp(-v1**2/(2*i*i))
                v3=np.mean(v2); g_FNR_.append(v3)
            self.GFNR.append(g_FNR_)
        
        
            g_FPR_=[] 
            for i in sigma:
                v1=np.array(self.d2)
                v1=v1*(v1<thresh2)
                v2=1-np.exp(-v1**2/(2*i*i))
                v3=np.mean(v2); g_FPR_.append(v3)
            self.GFPR.append(g_FPR_)
            
        # ravel lists
        self.GFNR=np.ravel(self.GFNR)
        self.GFPR=np.ravel(self.GFPR)
        
        if save:
            
            path=os.getcwd()
            dirr=path+'/'+foldername
            
            if not os.path.exists(dirr):
                os.mkdir(dirr)
            
            np.savetxt(dirr+'/GFNR.txt', self.GFNR)
            np.savetxt(dirr+'/GFPR.txt', self.GFPR)
            np.savetxt(dirr+'/stats.txt', [self.n_nodes_r,
                                           self.n_nodes_e,
                                           self.n_branches_r,
                                           self.n_branches_e])
                
    def plotDist(self, save=False, foldername=None):
                 
        try:
            import seaborn as sns
        except:
            print('To run this function, \'seaborn\' sould be installed.')
            return         
            
        sns.set_style('darkgrid')  
        
        if foldername:
            pass
        else:
            foldername=self.outputfolder
            
            
        plt.figure(figsize=(8.3,5.5))
        sns.kdeplot(self.d1, 
                label=r'$\mathbf{J}_{r}$ $\rightarrow$ $\mathbf{J}_{exp}$', 
                cut=0, marker='s', markevery=0.05, linewidth=2)   
            
        sns.kdeplot(self.d2, 
                    label=r'$\mathbf{J}_{e}$ $\rightarrow$ $\mathbf{J}_{real}$', 
                    cut=0, marker='8', markevery=0.05, linewidth=2) 
        plt.legend(fontsize=22)
        plt.ylabel('Probability', fontsize=20); plt.xlabel('$D$', fontsize=20) 
        plt.xlim(xmin=0 , xmax=80)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        
        if save:
            
            path=os.getcwd()
            dirr=path+'/'+foldername
            
            if not os.path.exists(dirr):
                os.mkdir(dirr)
                
            plt.savefig(dirr+'/dist.eps', format='eps', dpi=1000, transparent=True)
            plt.close()



    def matchC(self, sigma=10):
         
        ############################
        # match nodes in both graphs based on distance threshold
        ############################
               
    
        # REAL TO EXP
        self.matchG()
            
        self.d1C=np.array(self.d1)            
        self.idx1_pass=np.where(self.d1C<sigma)[0] #to find matched nodes that pass the condition
        self.idNodes_real_pass=np.array(self.idNodes_real)[self.idx1_pass]
        
        self.idx1_fail=np.where(self.d1C>sigma)[0] #to find matched nodes that fail the condition
        self.idNodes_real_fail=np.array(self.idNodes_real)[self.idx1_fail]
        
        #find mapping1
        self.mapping1=[[i,j] for i,j in zip(self.idNodes_real, self.idNodes_exp_m)]
               
        
        # REAL TO EXP
         
        self.d2C=np.array(self.d2)
        
        self.idx2_pass=np.where(self.d2C<sigma)[0] #to find matched nodes that pass the condition
        self.idNodes_exp_pass=np.array(self.idNodes_exp)[self.idx2_pass]
        
        self.idx2_fail=np.where(self.d2C>sigma)[0] #to find matched nodes that fail the condition
        self.idNodes_exp_fail=np.array(self.idNodes_exp)[self.idx2_fail]
        
        #find mapping2
        self.mapping2=[[i,j] for i,j in zip(self.idNodes_exp, self.idNodes_real_m)]
            

    def compareGraphs(self):
        
            # mapping of nodes ('shared') in real and exp graphs 
            self.shared_nodes1=dict()
            self.shared_nodes2=dict()
            for i in self.mapping1:
                for j in self.mapping2: 
                    if i[::-1]==j:
                        self.shared_nodes1[i[0]]=i[1]
                        self.shared_nodes2[i[1]]=i[0]


            #neigbours of shared nodes in real graph
            self.nbrs1=dict()
            for i in self.G_real_reduced.GetNodes():
                nbrs_=self.G_real_reduced.GetNeighbors(i)
                nbrs_=[j for j in nbrs_ if j in self.shared_nodes1.keys()]
                self.nbrs1[i]=nbrs_
                
            
            #neigbours of shared nodes in exp graph
            self.nbrs2=dict()
            for i in self.G_exp_reduced.GetNodes():
                nbrs_=self.G_exp_reduced.GetNeighbors(i)
                nbrs_=[j for j in nbrs_ if j in self.shared_nodes2.keys()]
                self.nbrs2[i]=nbrs_
            
            
            #connection between nodes to removed from real graph
            self.c_to_remove1=[]    
            for i in self.shared_nodes1.keys():
                
                nbrs_=self.nbrs1[i]
                nbrs_=[self.shared_nodes1[j] for j in nbrs_]                
                nbrs_t=self.nbrs2[self.shared_nodes1[i]]
                
                nds_=list(set(nbrs_).difference(set(nbrs_t)))
                nds=[self.shared_nodes2[j] for j in nds_]
                c_=[[i,j] for j in nds]
                if c_:
                    self.c_to_remove1.append(c_)    
            self.c_to_remove1=[j for i in self.c_to_remove1 for j in i] # ravel
                
            #connection between nodes to removed from exp graph        
            self.c_to_remove2=[]    
            for i in self.shared_nodes2.keys():
                nbrs_=self.nbrs2[i]
                nbrs_=[self.shared_nodes2[j] for j in nbrs_]
                
                nbrs_t=self.nbrs1[self.shared_nodes2[i]]
                
                nds_=list(set(nbrs_).difference(set(nbrs_t)))
                nds=[self.shared_nodes1[j] for j in nds_]
                c_=[[i,j] for j in nds]
                if c_:
                    self.c_to_remove2.append(c_)
            self.c_to_remove2=[j for i in self.c_to_remove2 for j in i] # ravel
        
            
            #decide the pathes and then the points to be removed from the origional real graph    
            self.pth_to_remove1=[]
            for i in self.branches1:
                if len(i)>=2:
                    endA=i[0]
                    endB=i[-1]
                    i.pop(0)
                    i.pop()
                    if [endA, endB] in self.c_to_remove1 or [endB, endA] in self.c_to_remove1:
                        self.pth_to_remove1.append(i)   
                    
            self.nds_to_remove1=list(set([j for i in self.pth_to_remove1 for j in i])) #ravel    
             
            self.Gcore_real=getCoreGraph(self.Gr.copy(), self.idNodes_real_fail)
            self.Gcompared_real=self.Gcore_real.copy()
            self.Gcompared_real.remove_nodes_from(self.nds_to_remove1)
            
            
            #decide the pathes and then the points to removed from the origional exp graph    
            self.pth_to_remove2=[]
            for i in self.branches2:
                if len(i)>=2:
                    endA=i[0]
                    endB=i[-1]
                    i.pop(0)
                    i.pop()
                    if [endA, endB] in self.c_to_remove2 or [endB, endA] in self.c_to_remove2:
                        self.pth_to_remove2.append(i)   
                    
            self.nds_to_remove2=list(set([j for i in self.pth_to_remove2 for j in i])) #ravel    
            
            self.Gcore_exp=getCoreGraph(self.Ge.copy(), self.idNodes_exp_fail)            
            self.Gcompared_exp=self.Gcore_exp.copy()
            self.Gcompared_exp.remove_nodes_from(self.nds_to_remove2)            
            

    def scoresC(self):

        sigma=self.sigma
        
        self.CFNR=[]
        self.CFPR=[]
        self.precision=[]
        
        for i in sigma:
            
            print('Calculate at sigma = '+str(i))
            self.matchC(sigma=i)
            self.compareGraphs()
            
            branches_tP=getBranches(self.Gcompared_exp.copy())
            tP=len(branches_tP)        
            fP=len(self.branches2)-tP       
            
            #
            
            branches_fN=getBranches(self.Gcompared_real.copy())
            fN=len(self.branches1)-len(branches_fN)        
            
            CFNR_=float(fN)/(fN+tP)        
            CFPR_=float(fP)/(fP+tP)  
            
            self.CFNR.append(CFNR_)
            self.CFPR.append(CFPR_)
            self.Precision=float(tP)/(tP+fP)
        
        #ravel lists
        self.CFNR=np.ravel(self.CFNR)
        self.CFPR=np.ravel(self.CFPR)
   
    def GetScores(self):
        
        self.scoresG()
        self.scoresC()
        self.Gcore_real=fixG(self.Gcore_real)
        self.Gcompared_real=fixG(self.Gcompared_real)
        self.Gcore_exp=fixG(self.Gcore_exp)
        self.Gcompared_exp=fixG(self.Gcompared_exp)
        
        return [self.GFNR, self.GFPR, self.CFNR, self.CFPR]
        
                       
            