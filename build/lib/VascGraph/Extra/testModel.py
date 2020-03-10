#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:17:07 2018

@author: rdamseh
"""

from graphContraction import *
import pandas as ps
from time import time


l=sio.loadmat('data/test_model.mat')['model']
l=l[:,:460,(0,20,51)]
s=np.shape(l)
s=np.array([256.0,256.0,3.0])/s
l=sc.ndimage.zoom(l,s)
#visStack(l, opacity=.5)
times={}

savepath='figures/param/'
models=[]
#init=[1000,20000,20000,20000,20000,20000]
#c=[8,8,4,32,8,8]
#eps=[1,1,1,1,0.5,1.5]

init=[20000]
c=[64]
eps=[1]



for i,j,k  in zip(init,c,eps):
    
    t0=time()
    model=graphContraction(label=l)
    model.generateGraph(init_nodes=i, connect=j)
    model.contractGraph(a=1, eps=k)
    model.refineGraph(ln=10, alpha=75)
    models.append(model)
    times['param_'+str(i)+'_'+str(j)+'_'+str(k)]=time()-t0






#### time #####
    
#df=ps.DataFrame(data=times, index=[0])
#df.to_csv(savepath+'times.csv')
###############





# VIS #####
camGraph=createCam(
        position = [132, 116, 560],
        focal_point = [132, 132, 45],
        view_angle = 30.0,
        view_up = [0, -1, 0],
        clipping_range = [524, 600]
        )

mlab.figure(size=(256,256), bgcolor=(1,1,1))
visStack(l, color=(.7,.7,.7), opacity=.5, mode='same')
visG(models[0].G_refined, radius=2, color=(.3,.3,.7), gylph_r=.1, gylph_c=(.7,.7,.3),
            jnodes_r=8, jnodes_c=(.9,.9,.3))     
setCam(camGraph)
############
mlab.savefig(savepath+'param_'+str(init)+'_'+str(c)+'_'+str(eps)+'.png',size=(1024,1024))



    
        