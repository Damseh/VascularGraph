#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:30:47 2018

@author: rdamseh
"""

from graphContraction2 import *

import skimage.io as skio


#from Bio import AlignIO
#from Bio import Phylo
#from Bio.Phylo.TreeConstruction import *

i=1 # snakes: 7, 22, 13  pliers: 7 tables:
path='data/sim/data56noisy2/'
path='data/tpm/seg/'
#path='/home/rdamseh/objects3d/snakesIm/'
#path='/home/rdamseh/objects3d/pliersIm/'
#path='/home/rdamseh/objects3d/tablesIm/'
#path='/home/rdamseh/objects3d/humansIm/'
#path='/home/rdamseh/objects3d/octopusIm/'

#path='/home/rdamseh/objects3d/cupsIm/'
#path='/home/rdamseh/objects3d/cupsIm/'


# mri image ############
#mra_filepath='/home/rdamseh/GraphContraction \
#(code3)/MRA/Data/Normal-002/MRA/Normal002-MRA.mha'
#skull_filepath='/home/rdamseh/GraphContraction \
#(code3)/MRA/Data/Normal-002/AuxillaryData/SkullStripped-T1-Flash.mha'
#tree_filepath='/home/rdamseh/GraphContraction \
#(code3)/MRA/Data/Normal-002/AuxillaryData/VascularNetwork.tre'
#
#
#
#mra= io.imread(mri_filepath, plugin='simpleitk')
#mra=sc.ndimage.zoom(mra, (0.8,0.5,0.5))
#skull=io.imread(skull_filepath, plugin='simpleitk')
#tree = Phylo.read(tree_filepath, 'nexus')
##################



# testmodel #################
l=sio.loadmat('data/test_model.mat')['model']
l=l[:,:460,(0,5,10,20,25,30,35,40)]
s=np.shape(l)
s=np.array([256.0,256.0,3.0])/s
l=sc.ndimage.zoom(l,s)
####################




filename=''+str(i)
try:
    seg=sio.loadmat(path+filename+'.mat')['seg']
    #seg=sio.loadmat(path+filename+'.im.mat')['data3d']
except:
    seg=readStack(path+filename+'.tif') 
    s={'seg':seg>0}

#seg=l

#seg=seg[0:250,0:250,0:250]

###
# best param
#speed=.1
#dis=.5
#deg=.5 ?? contadict with thr; there might be no need for it
#med=1
#thr=1 ?? Problem
#stop_thr=.005 as samplel =1 // stop_thr at sample =2
###


t0=time()
s=np.array(np.shape(seg))


G=graphContraction(label=seg)
G.generateGraph(sample=2)
G.contractGraph2(speed_param=.1, #[low_speed, high_speed]
                 dis_param=.5,
                 deg_param=0,
                 med_param=1,
                 thr=5,
                 n_iter=10,
                 stop_thr=.01)

G.refineGraph(diam_refine=5)
print('Time to generate graph: '+str(time()-t0))

g0=G.G_init
g=G.G_contracted.copy()
gg=G.G_refined
ggg=G.G_final

visStack(seg, opacity=.2)
#visG(fixG(g0), color=(0,0,1), radius=.05, gylph_r=.1)
#visG(fixG(g), radius=.1, gylph_r=.5, jnodes_r=1, jnodes_c=(0,0,1))
visG(fixG(gg), radius=.2, color=(0,0,1), gylph_r=.5, jnodes_r=1, jnodes_c=(1,0,1))


visStack(seg, opacity=.2)
visG(fixG(ggg), radius=.2, gylph_r=.5, jnodes_r=1, jnodes_c=(0,0,1), diam=True)


j,_=findNodes(gg)
print(len(j))