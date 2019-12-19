#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:02:23 2018

@author: rdamseh
"""

from graphContraction import *


if __name__=='__main__':

    #param =(eps=), (eps=1, portion=.3), (eps=2, portion=.1), (eps=2, portion=.03)
    #(eps=.7, portion=.7), 
    
    #(eps=2, portion=.035) tpm , ln=10, alpha=50
    #(eps=2, portion=.035) tpm noisy , ln=10, alpha=100
    
    #(eps=1, portion=.25) sim noise-free , ln=10, alpha=25
    #(eps=1, portion=.25) sim noise , ln=10, alpha=25
    
    #(eps=.75, portion=.75) sim noise (b=6) , ln=10, alpha=25


    #(eps=.75, portion=.5, connect=8) natural , ln=10, alpha=25
    #(eps=.75, portion=.05, connect=6) teddy  , ln=10, alpha=25
 
    
    n=[4]
    
    for i in n:
    

        # 
        camGraph=createCam(
position = [225.80682866701738, 154.6150093768255, -78.2645722876704],
focal_point = [43.65290081123969, 41.35379855500724, 53.6987412268675],
view_angle = 30.0,
view_up = [0.2751685540032653, -0.8836968091570157, -0.37863203294684555],
clipping_range = [81.24152987070005, 445.7441883580216]
                )
        
        path='data/sim/data16noisy1/'
        #path='data/tpm/seg/seg'
        #path='data/tpm/noisy/seg/'
        filename=str(i)+'morph'
        
        try:
            seg=sio.loadmat(path+filename+'.mat')['seg']
        except:
            seg=readStack(path+filename+'.tif') 
            s={'seg':seg>0}
            sio.savemat(path+filename+'.mat', s,  do_compression=True)
        #seg=sio.loadmat('/home/rdamseh/objects3d/teddyIm/b5.im.mat')['data3d']
    
        G=graphContraction(label=seg, animate='animation/', camGraph=camGraph)
        #G=graphContraction(label=seg)
        G.generateGraph(connect=8, portion=.45)
        G.contractGraph(eps=.9)
        G.refineGraph(ln=10, alpha=30)
        

        g=G.G_refined
        
        #gg=readPAJEK(path+'basicgraphs/'+str(i)+'.pajek')
        #gg=adjustGraph(gg, flip=[0,0,0], switch=[0,1,0])
        
        mlab.close(all=True)
        mlab.figure(bgcolor=(1,1,1), size=(1024,1024)) 
        visStack(seg, color=(.7,.7,.7), opacity=.3, mode='same')
        visG(g, radius=.7, color=(.3,.3,.7), gylph_r=2, gylph_c=(.7,.7,.3),
             jnodes_r=4, jnodes_c=(.7,.3,.3))
        
        setCam(camGraph)
        mlab.savefig('animation/'+'final1.png', size=(1024,1024))

#        
        #visStack(seg, color=(.7,.7,.7), opacity=.1)
        #visG(gg, radius=2, color=(.7,.3,.3), gylph_r=5, gylph_c=(.7,.7,.3))
        
#        gmesh=getGraphfromCGAL('/home/rdamseh/skeleton/MeanCurvatureSkeleton/build-skltn-Desktop-Debug/vertices.cgal',
#                               '/home/rdamseh/skeleton/MeanCurvatureSkeleton/build-skltn-Desktop-Debug/edges.cgal')
#        gmesh=adjustGraph(gmesh.copy(), flip=(0,1,0), switch=(0,1,0)) 
#        nx.write_pajek(gmesh, path+'gmesh'+str(i))
#        m=readOFF(path+str(i)+'.off')
#        m=adjustMesh(m, flip=(0,1,0), switch=(0,1,0))
#        visMesh(m, opacity=.1)
#        visG(gmesh, radius=1, color=(0,0,.7), gylph_r=.1, gylph_c=(1,1,0))
        
        
        
        
        
        
        
        
        
        