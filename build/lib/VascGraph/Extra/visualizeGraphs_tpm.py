#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:02:23 2018

@author: rdamseh
"""

from graphContraction import *


if __name__=='__main__':
  
    # 6,7,13
    n=[1]
    savepath='figures/tpm/'
    
    for i in n:
        
        camGraph=createCam(
        position = [1388.0089234463396, -310.47389220413436, -267.76942578853027],
        focal_point = [244.2935362616005, 275.3680360075078, 290.272624765627],
        view_angle = 30.0,
        view_up = [-0.32931945680359265, 0.22951926239313705, -0.9158982495676564],
        clipping_range = [522.8495173652159, 2427.0787396080887]
        )
        
        path='data/tpm/seg/'
        path_gt='data/tpm/seg/'
        path_raw='data/tpm/seg/raw/'

        
        filename=str(i)+''
        
        # read seg
        try:
            seg=sio.loadmat(path+filename+'.mat')['seg']
        except:
            seg=readStack(path+filename+'.tif') 
            s={'seg':seg>0}
            sio.savemat(path+filename+'.mat', s,  do_compression=True)
        
        # read raw
        try:
            raw=sio.loadmat(path_raw+filename+'.mat')['raw']
        except:
            raw=readStack(path_raw+filename+'.tif') 
            raw=np.rollaxis(raw,0, 3)
            raw={'raw':raw}
            sio.savemat(path_raw+filename+'.mat', raw,  do_compression=True)
            
            
        # read graphs
        g=readPAJEK(path+'mygraphs/'+str(i)+'.pajek') # mygraph
        g=rescaleG(g)
        g=adjustGraph(g, flip=[1,0,0], switch=[1,0,0])


        ggt=readPAJEK(path_gt+'groundtruth/'+str(i)+'.pajek') # groundtruth graph
        ggt=rescaleG(ggt)
        ggt=adjustGraph(ggt, flip=[0,0,0], switch=[0,0,0])
        
        gb=readPAJEK(path+'basicgraphs/'+str(i)+'.pajek') # basic graph
        gb=rescaleG(gb)
        gb=adjustGraph(gb, flip=[0,1,0], switch=[0,0,0])
        
        gmesh=readPAJEK(path+'meshgraphs/'+str(i)+'.pajek')
        gmesh=rescaleG(gmesh)
        
        ###############################################
        #vis gtigraph 
        mlab.figure(bgcolor=(1,1,1))
        visG(ggt, radius=3, color=(.7 ,.7, .7), gylph_r=.1, gylph_c=(.7,.7,.3),
             jnodes_r=5, jnodes_c=(.9,.9,.3))     
        setCam(camGraph)
        #mlab.savefig(savepath+'tpmgt.png',size=(1024,1024))
        
        #vis bascigraph
        mlab.figure(bgcolor=(1,1,1))
        visG(gb, radius=3, color=(.7,.2,.2), gylph_r=.1, gylph_c=(.7,.7,.3),
            jnodes_r=5, jnodes_c=(.9,.9,.3))     
        setCam(camGraph)      
        #mlab.savefig(savepath+'tpmbsc.png',size=(1024,1024))
        
        #vis meshgraph
        mlab.figure(bgcolor=(1,1,1))
        visG(gmesh, radius=3, color=(.2,.8,.2), gylph_r=.1, gylph_c=(.7,.7,.3),
            jnodes_r=5, jnodes_c=(.9,.9,.3))     
        setCam(camGraph)
        #mlab.savefig(savepath+'tpmmsh.png',size=(1024,1024))
        
        #vis mygraph
        mlab.figure(bgcolor=(1,1,1))
        visG(g, radius=3, color=(.2,.2,.7), gylph_r=.1, gylph_c=(.7,.7,.3),
            jnodes_r=5, jnodes_c=(.9,.9,.3))     
        setCam(camGraph)
        #mlab.savefig(savepath+'tpmme.png',size=(1024,1024))
        
        ##############################3
        
        
        
        
        ### plot disconnected components in g basic ###
        
        
        graphs=getDisConnGraphs(gb)
        #vis 
        mlab.figure(bgcolor=(1,1,1))
        visG(gb, radius=3, color=(.7,.2,.2), gylph_r=.1, gylph_c=(.7,.7,.3),
            jnodes_r=5, jnodes_c=(.9,.9,.3))     
        
        for idx, i in enumerate(graphs):           
            if idx>0:
                visG(i, radius=5, color=(1,1,0))     
                
        setCam(camGraph)  
        #mlab.savefig(savepath+'tpmbsc.png',size=(1024,1024))