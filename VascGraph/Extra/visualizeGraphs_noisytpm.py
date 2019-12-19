#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:02:23 2018

@author: rdamseh
"""

from graphContraction import *


if __name__=='__main__':
  
    # 6,7,13
    n=[14]
    savepath='figures/noisytpm/tpmnoisy'

    for i in n:
    

        # 
        camGraph=createCam(
            position = [-1110.0080188633806, 84.27543333489109, 131.11554893684223],
            focal_point = [36.04639753379627, 246.41210034947252, 249.08968133565236],
            view_angle = 30.0,
            view_up = [0.10153604100641063, 0.0061451761216366006, -0.9948128814943938],
            clipping_range = [959.8491763456833, 1422.5779423069987]
            )
        
        camGraphZoom=createCam(   
            position = [-296.412425539163, 66.9340911919076, 69.40828902356441],
            focal_point = [68.75583029803511, 118.59582998359542, 106.99848329846712],
            view_angle = 30.0,
            view_up = [0.10153604100641063, 0.0061451761216366006, -0.9948128814943938],
            clipping_range = [175.0285991969925, 617.9386637151589]
            )
        
        path='data/tpm/noisy/seg/'
        
        filename=str(i)+''
        try:
            seg=sio.loadmat(path+filename+'.mat')['seg']
        except:
            seg=readStack(path+filename+'.tif') 
            s={'seg':seg>0}
            sio.savemat(path+filename+'.mat', s,  do_compression=True)
        
        #############
        #read raw
        path_raw='data/tpm/noisy/raw/'
        try:
            raw=sio.loadmat(path_raw+filename+'.mat')['seg']
        except:
            raw=readStack(path_raw+filename+'.tif') 
            r={'raw':raw}
            sio.savemat(path_raw+filename+'.mat', r,  do_compression=True)
        raw_mip=makeMIP(raw)
        plt.imsave(path_raw+'mip'+filename,raw_mip[0].T, cmap='plasma')        
        ###############
        
        # read graphs
        g=readPAJEK(path+'mygraphs/'+str(i)+'.pajek') # mygraph
        

        gb=readPAJEK(path+'basicgraphs/'+str(i)+'.pajek') # basic graph
        gb=adjustGraph(gb, flip=[0,0,0], switch=[0,1,0])
        
        
        #vis bascigraph    
        visStack(seg, color=(.7,.7,.7), opacity=.25)
        visG(gb, radius=2, color=(.7,.2,.2), gylph_r=.1, gylph_c=(.7,.7,.3),
            jnodes_r=5, jnodes_c=(.9,.9,.3))     
        setCam(camGraph)      
        mlab.savefig(savepath+'bsc.png',size=(1024,1024))
        
        #vis mygraph
        visStack(seg, color=(.7,.7,.7), opacity=.25)
        visG(g, radius=2, color=(.2,.2,.7), gylph_r=.1, gylph_c=(.7,.7,.3),
            jnodes_r=5, jnodes_c=(.9,.9,.3))     
        setCam(camGraph)
        mlab.savefig(savepath+'me.png',size=(1024,1024))

        #vis bascigraph zoom    
        visStack(seg, color=(.7,.7,.7), opacity=.25)
        visG(gb, radius=2, color=(.7,.2,.2), gylph_r=.1, gylph_c=(.7,.7,.3),
            jnodes_r=5, jnodes_c=(.9,.9,.3))     
        setCam(camGraphZoom)      
        mlab.savefig(savepath+'bsczoom.png',size=(1024,1024))
        
        #vis mygraph zoom
        visStack(seg, color=(.7,.7,.7), opacity=.25)
        visG(g, radius=2, color=(.2,.2,.7), gylph_r=.1, gylph_c=(.7,.7,.3),
            jnodes_r=5, jnodes_c=(.9,.9,.3))     
        setCam(camGraphZoom)
        mlab.savefig(savepath+'mezoom.png',size=(1024,1024))
        
        
#        gmesh=getGraphfromCGAL('/home/rdamseh/skeleton/MeanCurvatureSkeleton/build-skltn-Desktop-Debug/vertices.cgal',
#                               '/home/rdamseh/skeleton/MeanCurvatureSkeleton/build-skltn-Desktop-Debug/edges.cgal')
#        gmesh=adjustGraph(gmesh.copy(), flip=(0,1,0), switch=(0,1,0)) 
#        nx.write_pajek(gmesh, path+'gmesh'+str(i))
#        m=readOFF(path+str(i)+'.off')
#        m=adjustMesh(m, flip=(0,1,0), switch=(0,1,0))
#        visMesh(m, opacity=.1)
#        visG(gmesh, radius=1, color=(0,0,.7), gylph_r=.1, gylph_c=(1,1,0))
        
        
        
          
        
        
        
        
        