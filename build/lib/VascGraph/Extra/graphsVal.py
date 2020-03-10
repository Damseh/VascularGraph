#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:27:18 2018

@author: rdamseh
"""

from util_validation import *
import scipy as sc
import seaborn as sns
sns.set_style('darkgrid')


def setC(s='full'):
    
    if s=='full':
        e=mlab.get_engine()
        c=e.current_scene
        c.scene.camera.position = [1508.0806790757697, -75.76570871449209, -246.80378086524414]
        c.scene.camera.focal_point = [275.9113889043341, 290.74905627194926, 310.09149593802437]
        c.scene.camera.view_angle = 30.0
        c.scene.camera.view_up = [-0.39322212027637654, 0.07091714343590497, -0.9167044904942061]
        c.scene.camera.clipping_range = [586.1793691504713, 2399.2338745098964]
        c.scene.camera.compute_view_plane_normal()
        c.scene.render()
    
    if s=='mag':
        e=mlab.get_engine()
        c=e.current_scene
        c.scene.camera.position = [228.2594962041794, 245.50442377609815, 193.67886145992924]
        c.scene.camera.focal_point = [228.2594962041794, 245.50442377609815, 248.55941772460938]
        c.scene.camera.view_angle = 30.0
        c.scene.camera.view_up = [0.0, 1.0, 0.0]
        c.scene.camera.clipping_range = [0.5371343542211496, 537.1343542211496]
        c.scene.camera.compute_view_plane_normal()
        c.scene.render()



def pltDistance(d1, d2, linestyle='-', figure=True, tag='', savepath=None):
    
    if figure:
        plt.figure(figsize=(8.3,5.5))
        
    sns.kdeplot(d1, 
            label=r'$\mathbf{J}_{r}$ $\rightarrow$ $\mathbf{J}_{e}$ ('+tag+')', 
            cut=0, linestyle=linestyle, markevery=0.05, linewidth=2, color=(0,.4,.6))   
        
    sns.kdeplot(d2, 
                label=r'$\mathbf{J}_{e}$ $\rightarrow$ $\mathbf{J}_{r}$ ('+tag+')', 
                cut=0, linestyle=linestyle, markevery=0.05, linewidth=2, color=(.5,.5,0)) 
    plt.legend(fontsize=22)
    plt.ylabel('Probability', fontsize=20); plt.xlabel('$D$', fontsize=20) 
    plt.xlim(xmin=0 , xmax=60)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    
    if savepath:
        plt.savefig(savepath+'dist.eps', format='eps', dpi=1000, transparent=True)


if __name__=='__main__':


    path='graphs/'
    names1=['gr1', 'gr2', 'gr3', 'gr4', 'gr5', 'gr6']    
    names2=['g1', 'g2', 'g3', 'g4', 'g5', 'g6']        
    names3=['gbasic1', 'gbasic2', 'gbasic3', 'gbasic4', 'gbasic5', 'gbasic6']        


    graphs1=[rescaleG(readPAJEK(path+i+'.pajek')) for i in names1]
    graphs2=[rescaleG(readPAJEK(path+i+'.pajek')) for i in names2]
    #graphs2=[getFullyConnected(i) for i in graphs2 ]


    graphs33=[rescaleG(readPAJEK(path+i+'.pajek')) for i in names3]
    #graphs3=[getFullyConnected(i) for i in graphs33 ]
    graphs3=[adjustGraph(i, flip=[0,1,0]) for i in graphs3]
    

    savepath='graphs/figures/'

    
    ##########
    ### vis distance matching (for geaometric validation)
    ##########
        
        
    orig_nodes=[i.number_of_nodes() for i in graphs33]
    nodes=[i.number_of_nodes() for i in graphs3]
    jnodes=[len(findNodes(i, j_only=True)[0]) for i in graphs3]
    percent=[float(i)/j for i , j in zip(nodes, orig_nodes)]
    
       
    # create validation objects + get score
    valsbasic=[validate(i, j, sigma=[60], rescale=True, middle=64)
                for i, j in zip(graphs1, graphs3)]   
    sbasic=[i.scores() for i in valsbasic]
    sbasic=np.array(sbasic)


    valsmesh=[validate(i, j, sigma=[60], rescale=True, middle=64)
                for i, j in zip(graphs1, graphs2)]
    smesh=[i.scores() for i in valsmesh]
    smesh=np.array(smesh)
 
    # average scores
    avmesh=np.mean(smesh, axis=0)
    avbasic=np.mean(sbasic, axis=0)   
    varmesh=np.var(smesh, axis=0)
    varbasic=np.var(sbasic, axis=0)
    
    
    # get and plot matching 
    d=5
    
    d1basic = [i.d1 for i in valsbasic]
    d2basic = [i.d2 for i in valsbasic]
    d1mesh = [i.d1 for i in valsmesh]
    d2mesh = [i.d2 for i in valsmesh]
       
    d1b=[]
    [d1b.extend(i) for i in d1basic]
    d2b=[]
    [d2b.extend(i) for i in d2basic]
    d1m=[]
    [d1m.extend(i) for i in d1mesh ]
    d2m=[]
    [d2m.extend(i) for i in d2mesh ]

    #pltDistance(d1=d1basic[d], d2=d2basic[d], linestyle='--', tag='3D thining')
    #pltDistance(d1=d1mesh[d], d2=d2mesh[d], figure=False, tag='Proposed')

    pltDistance(d1=d1b, d2=d2b, linestyle='--', tag='3D thining')
    pltDistance(d1=d1m, d2=d2m, figure=False, tag='Proposed', savepath=savepath)






















  
    