#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:02:23 2018

@author: rdamseh
"""

from graphContraction import *


if __name__=='__main__':
  
    # 6,7,13
    n=[9]
    savepath='figures/sim/noisy'
    for i in n:
    

        # 
        camGraph=createCam(
            position = [159.2069935372582, 161.0351370984544, 155.100711784022],
            focal_point = [64.25937939734665, 66.08752295854283, 60.15309764411047],
            view_angle = 30.0,
            view_up = [0.0, 0.0, 1.0],
            clipping_range = [47.91684726155867, 311.54714555598974]
                )
        
        camGraph=createCam(position = [87.32707596364841, 281.2786086155121, -103.22450299199687],
        focal_point = [48.77736135418737, 52.489267552872754, 53.95565164357538],
        view_angle = 30.0,
        view_up = [0.37342900110531163, 0.4817229376396192, 0.7927760039792734],
        clipping_range = [124.82013596839533, 476.56475081502026])
        
        path='data/sim/data56noisy2/'
        #path='data/sim/data56/'

        #path='data/sim/data16noisy2/'
        #path='data/sim/data16/'
        
        
        path_gt='data/sim/data56/'
        #path_gt='data/sim/data16/'

        #path='data/tpm/seg/seg'
        #path='data/tpm/noisy/seg/'
        
        filename=str(i)+''
        try:
            seg=sio.loadmat(path+filename+'.mat')['seg']
        except:
            seg=readStack(path+filename+'.tif') 
            s={'seg':seg>0}
            sio.savemat(path+filename+'.mat', s,  do_compression=True)
        
        
        # read graphs
        g=readPAJEK(path+'mygraphs/'+str(i)+'.pajek') # mygraph
        
        ggt=readPAJEK(path_gt+'groundtruth/'+str(i)+'.pajek') # groundtruth graph
        ggt=adjustGraph(ggt, flip=[0,0,0], switch=[0,1,0])
        
        gb=readPAJEK(path+'basicgraphs/'+str(i)+'.pajek') # basic graph
        gb=adjustGraph(gb, flip=[0,0,0], switch=[0,1,0])
        
        gmesh=readPAJEK(path+'meshgraphs/'+str(i)+'.pajek')
        gmesh=adjustGraph(gmesh.copy(), flip=(0,1,0), switch=(0,1,0)) 
        
        #vis gtigraph    
        visStack(seg, color=(.7,.7,.7), opacity=.5)
        visG(ggt, radius=2, color=(.7 ,.7, .7), gylph_r=.1, gylph_c=(.7,.7,.3),
             jnodes_r=5, jnodes_c=(.9,.9,.3))     
        setCam(camGraph)
        mlab.savefig(savepath+'gt.png',size=(1024,1024))
        
        #vis bascigraph    
        visStack(seg, color=(.7,.7,.7), opacity=.5)
        visG(gb, radius=2, color=(.7,.2,.2), gylph_r=.1, gylph_c=(.7,.7,.3),
            jnodes_r=5, jnodes_c=(.9,.9,.3))     
        setCam(camGraph)      
        mlab.savefig(savepath+'bsc.png',size=(1024,1024))
        
        #vis meshgraph
        visStack(seg, color=(.7,.7,.7), opacity=.5)
        visG(gmesh, radius=2, color=(.2,.8,.2), gylph_r=.1, gylph_c=(.7,.7,.3),
            jnodes_r=5, jnodes_c=(.9,.9,.3))     
        setCam(camGraph)
        mlab.savefig(savepath+'msh.png',size=(1024,1024))
        
        #vis mygraph
        visStack(seg, color=(.7,.7,.7), opacity=.5)
        visG(g, radius=2, color=(.2,.2,.7), gylph_r=.1, gylph_c=(.7,.7,.3),
            jnodes_r=5, jnodes_c=(.9,.9,.3))     
        setCam(camGraph)
        mlab.savefig(savepath+'me.png',size=(1024,1024))


        
#        gmesh=getGraphfromCGAL('/home/rdamseh/skeleton/MeanCurvatureSkeleton/build-skltn-Desktop-Debug/vertices.cgal',
#                               '/home/rdamseh/skeleton/MeanCurvatureSkeleton/build-skltn-Desktop-Debug/edges.cgal')
#        gmesh=adjustGraph(gmesh.copy(), flip=(0,1,0), switch=(0,1,0)) 
#        nx.write_pajek(gmesh, path+'gmesh'+str(i))
#        m=readOFF(path+str(i)+'.off')
#        m=adjustMesh(m, flip=(0,1,0), switch=(0,1,0))
#        visMesh(m, opacity=.1)
#        visG(gmesh, radius=1, color=(0,0,.7), gylph_r=.1, gylph_c=(1,1,0))
        
        
        
          
        
        
        
        
        