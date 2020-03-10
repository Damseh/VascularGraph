#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:02:23 2018

@author: rdamseh
"""

from graphContraction import *


if __name__=='__main__':

    
    #(eps=2, portion=.035, conn=8) tpm , ln=10, alpha=50
    #(eps=2, portion=.035, conn=8) tpm noisy , ln=10, alpha=100
    
    #(eps=.75, portion=.5, conn=6) sim , ln=10, alpha=25
    #(eps=.75, portion=.5, conn=6) sim noise1 , ln=10, alpha=25
    #(eps=.75, portion=.5, conn=6) sim noise2 , ln=10, alpha=25
    #(eps=.75, portion=.5, conn=6) sim noise3 , ln=10, alpha=25

    
    #(eps=.75, portion=.5, connect=8) natural , ln=10, alpha=25
    #(eps=.75, portion=.05, connect=6) teddy  , ln=10, alpha=25
 
    
    n=[1,2,3,4,5,6,7,8,9,10]
    times=[]
    for i in n:
        
        #path='data/sim/data56/'
        path='data/sim/data56noisy2/'
        #path='data/sim/data56/'
        #path='data/tpm/seg/seg'
        #path='data/tpm/noisy/seg/'
        filename=str(i)+''
        
        try:
            seg=sio.loadmat(path+filename+'.mat')['seg']
        except:
            seg=readStack(path+filename+'.tif') 
            s={'seg':seg>0}
            sio.savemat(path+filename+'.mat', s,  do_compression=True)
            
        #seg=sio.loadmat('/home/rdamseh/objects3d/teddyIm/b5.im.mat')['data3d']
        #G=graphContraction(label=seg, animate='animation/', camGraph=camGraph)
        t0=time()
        
        G=graphContraction(label=seg)
        G.generateGraph(connect=8, portion=.5)
        G.contractGraph(eps=.75)
        G.refineGraph(ln=10, alpha=25)
        
        times.append(time()-t0)
        
        g1=G.G_init
        g2=G.G_contracted
        g=G.G_refined
        
        #nx.write_pajek(g, path+'mygraphs/'+str(i)+'.pajek')    
        
        g=readPAJEK(path+'mygraphs/'+str(i)+'.pajek')
        visStack(seg, opacity=.2)
        visG(g, radius=1, color=(0,0,.7), gylph_r=3, gylph_c=(1,1,0))

        
        
        
        
        
        