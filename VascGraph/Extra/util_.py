#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:41:32 2017

@author: Rafat Damseh
"""

import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import time
from mayavi import mlab
import h5py
import scipy.ndimage as ndim
import os
matplotlib.use('AGG')
from keras.layers import (Activation, Convolution3D, Dense, Dropout, Flatten,
                          MaxPooling3D)
from keras.models import Sequential
from tqdm import tqdm
from mayavi.sources.api import ArraySource
from mayavi.filters.api import Stripper, Tube, WarpScalar, PolyDataNormals, Contour, UserDefined
from mayavi.filters.set_active_attribute import SetActiveAttribute
from mayavi.modules.api import IsoSurface, Surface, Outline, Text3D, Glyph
from mayavi.tools.pipeline import line_source 
from scipy.spatial import ConvexHull as fc
import skimage.morphology as morph1
import scipy.ndimage.morphology as morph2
from skimage.morphology import skeletonize_3d as sk3d
import scipy.ndimage.filters as filt
from sklearn.cluster import DBSCAN
import matplotlib
import networkx as nx
from PIL import Image
import itertools as it
import scipy as sp

def visPredict(v):    
    indices=np.where(v>0)
    x1,x2,x3=indices[0],indices[1],indices[2]   
    d=mlab.pipeline.scalar_scatter(x1,x2,x3)
    g=mlab.pipeline.glyph(d)
    
    
def visBatch(v):
    src = mlab.pipeline.scalar_field(v)
    mlab.pipeline.iso_surface(src, contours=[1, 0])

def visVolume(v):
    src = mlab.pipeline.scalar_field(v)
    mlab.pipeline.volume(src)
    
def getDist(v):
    
    l1=[v[n,:,:] for n in range(np.shape(v)[0])] #Z-XY
    l2=[v[:,n,:] for n in range(np.shape(v)[1])] #X-ZY
    l3=[v[:,:,n] for n in range(np.shape(v)[2])] #Y-ZX
    
    d1=np.array([morph2.distance_transform_edt(l1[i]) for i in range(len(l1))])
    d1=d1.transpose((0,1,2))
    d2=np.array([morph2.distance_transform_edt(l2[i]) for i in range(len(l2))])
    d2=d2.transpose((1,0,2))
    d3=np.array([morph2.distance_transform_edt(l3[i]) for i in range(len(l3))])
    d3=d3.transpose((1,2,0))
    
    d_=np.maximum(d1,d2);d=np.maximum(d_,d3)
    return d  

def getDist(v):
    
    l1=[v[n,:,:] for n in range(np.shape(v)[0])] #Z-XY
    l2=[v[:,n,:] for n in range(np.shape(v)[1])] #X-ZY
    l3=[v[:,:,n] for n in range(np.shape(v)[2])] #Y-ZX
    
    d1=np.array([morph2.distance_transform_edt(l1[i]) for i in range(len(l1))])
    d1=d1.transpose((0,1,2))
    d2=np.array([morph2.distance_transform_edt(l2[i]) for i in range(len(l2))])
    d2=d2.transpose((1,0,2))
    d3=np.array([morph2.distance_transform_edt(l3[i]) for i in range(len(l3))])
    d3=d3.transpose((1,2,0))
    
    d_=np.maximum(d1,d2);d=np.maximum(d_,d3)
    return d  

    
def getDataset(path, scale):
    
    X=[]
    Y=[]

    f=h5py.File(path,'r')
    num_P=np.array(f['numberOfPositive'])
    num_N=np.array(f['numberOfNegative'])

    
    if scale:
        
                
        ind=np.array(range(num_P[scale-1])).astype('int')
        np.random.shuffle(ind)
        s=scale
        for i in ind:      
            d_name=['pos_scale'+str(s)+'_'+str(i)]           
            d_pos=np.array(f[d_name[0]])
            X.append(d_pos)
            Y.append(0)
        
        ind=np.array(range(num_N[scale-1])).astype('int')
        np.random.shuffle(ind)
        s=scale
        for i in ind:      
            d_name=['neg_scale'+str(s)+'_'+str(i)]           
            d_neg=np.array(f[d_name[0]])
            X.append(d_neg)
            Y.append(1)
        
            
    else:
        
        for s in tqdm(range(len(num_d))):
            ind=np.array(range(num_d[s])).astype('int')
            np.random.shuffle(ind)
          
            for i in ind:      
                d_name=['pos_scale'+str(s+1)+'_'+str(i),
                        'neg_scale'+str(s+1)+'_'+str(i)]    
            
                d_pos=np.array(f[d_name[0]])
                d_neg=np.array(f[d_name[1]])
                z=np.array(np.shape(d_pos)).astype('float')
                z=np.divide(np.array([16,32,32]),z)  
                # check if sample has the targeted size
                if True in [t != 1 for t in z]: 
                    d_pos=ndim.zoom(d_pos,z)
                    d_neg=ndim.zoom(d_neg,z)
                    X.append(d_pos);X.append(d_neg)
                    Y.append(0);Y.append(1)
    
    f.close()
    X=np.array(X[:])
    Y=np.array(Y[:])
    return X, Y


#Define a model 
def getModel(args):
    
    model = Sequential()
    model.add(Convolution3D(32, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3,
                            batch_input_shape=((1,32,32,32,1)), border_mode='same',
                            activation='relu'))
    
    model.add(Convolution3D(32, kernel_dim1=3, kernel_dim2=3,
                            kernel_dim3=3, border_mode='same', activation='relu'))
    
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Convolution3D(64, kernel_dim1=3, kernel_dim2=3,
                            kernel_dim3=3, border_mode='same', activation='relu'))
    
    model.add(Convolution3D(64, kernel_dim1=3, kernel_dim2=3,
                            kernel_dim3=3, border_mode='same', activation='relu'))
    
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(args.nb_classes, init='normal'))
    model.add(Activation('softmax'))
    return model




# creat plots or results
def plotHistory(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


# save results
def saveHistory(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))   

def cameraPosition(scene):   
    
    scene.scene.camera.position = [-567.41776413974469, 1633.7556647598549, 1363.3615455804083]
    scene.scene.camera.focal_point = [561.47961291021682, 626.13187551647002, -12.72703501800356]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.47290363159850934, -0.47961203996493962, 0.73914440154925776]
    scene.scene.camera.clipping_range = [929.45157879168619, 3476.2045332275461]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()

def addText(module,text,position):
    
    text3d=module
    text3d.text = text
    text3d.scale=[20,20,20]
    text3d.position=np.add(position,[0,0,0])



def visStack(v, opacity=.5, color=(1,0,0), mode=''):
    
    if mode!='same':
        mlab.figure(bgcolor=(1,1,1)) 


    s=mlab.get_engine() # Returns the running mayavi engine.
    scene=s.current_scene # Returns the current scene.
    scene.scene.disable_render = True # for speed
    
    origion=[0,0,0]
    label= 'Segmentation'
    
    A=ArraySource(scalar_data=v)
    A.origin=np.array(origion)
    D=s.add_source(A)# add the data to the Mayavi engine
    #Apply gaussain 3d filter to smooth visualization
    
    
    F=mlab.pipeline.user_defined(D, filter='ImageGaussianSmooth')
    F.filter.set_standard_deviation(0,0,0)
    contour = Contour()
    s.add_filter(contour)
    
    smooth = mlab.pipeline.user_defined(contour, filter='SmoothPolyDataFilter')
    smooth.filter.number_of_iterations = 1
    smooth.filter.relaxation_factor = 0  

    surface=Surface()
    s.add_module(surface)
    
    surface.module_manager.scalar_lut_manager.lut_mode = u'coolwarm'
    surface.module_manager.scalar_lut_manager.reverse_lut = True
    surface.actor.property.opacity = opacity
    surface.actor.mapper.scalar_visibility=False
    surface.actor.property.color = color #color
    
    return surface


def readStack(name, l=None, rgb=None):
    
    data=np.array(io.MultiImage(name))
    
    if l: 
        if len(np.shape(data))>3:
            data=(data[0,0:l,:,:,0])
        else:
            data=(data[0:l,:,:])
 
    else:
        if rgb:
            data=(data[0,:,:,:,:])           
        else:
                
            if len(np.shape(data))==5:
                data=(data[0,:,:,:,0])
            else:
                if len(np.shape(data))==4:
                    data=(data[0,:,:,:])
                else:                
                    data=(data[:])
    return data

class getSegData:
        
    pad=[64,64,64]
    posBatches=list()#to store positive samples
    negBatches=list()#to store negative samples
    nBatches=0.0
    thr=[.3, .7]
    indP=list()  
    indN=list()
    pSize=[[8,8,16,8,16,8,16],[16,32,32,64,64,128,128],[16,32,32,64,64,128,128]]   
    #    pSize=[[16],[32],[32]]  
        
    def __init__(self,segData):
        
        self.Data=segData
    
    
    def process1(self, E_param, AE_param=.01, padding=True):
        self.E_param=E_param
        self.AE_param=AE_param
        
        if padding:
            self.Data= np.pad(self.Data, ((self.pad[0],self.pad[0]),(self.pad[1],self.pad[1]),(self.pad[2],self.pad[2])),
                              mode='constant', constant_values = 0)
        self.segData=self.Data
        self.d1,self.d2,self.d3=np.shape(self.segData)
       
        self.procSeg()
    
    def process2(self):
        
        self.getSkl()
        self.getDist()
        self.getKeys()

        # indicies
        indices=np.where(self.sklData>0)
        self.indSkl=np.array([indices[0],indices[1],indices[2]]).T                         
        
        indices=np.where(self.keysData>0)
        indices=np.array([indices[0],indices[1],indices[2]]).T
        self.indKeys=getCluster(indices, eps=1, min_samples=1).astype('int')
#        self.indKeys=indices


    
    ### First Method ###        
    
    def checkV(self, ind):

        """
        This function returns a batch if it does satisfy the criterion.
        It operates by masking the batch at the selected index with a sphere at different sizes, once
        a batch at certain size pass the criterion, the function return it and stop
        """  
        
        i= int(self.distDataF[ind[0],ind[1],ind[2]]*1.5)
            
#        batch=self.segData[ind[0]-i:ind[0]+i+1 , ind[1]-i:ind[1]+i+1, ind[2]-i:ind[2]+i+1] 
#        sphere=morph1.ball(i).astype('bool')
#        v=np.bitwise_and(sphere, batch)
        
        batch=self.distDataF[ind[0]-i:ind[0]+i+1 , ind[1]-i:ind[1]+i+1, ind[2]-i:ind[2]+i+1] 
        sphere=morph1.ball(i)
        v=sphere*batch
        
        return v

    def appendBatch(self, B):
    
        """
        This function save the extracted batch.
        """
        
        self.Batches.append(B)           


  
    def getBatches1(self, outS=[32.0,32.0,32.0]):
    
        """
        This function extract positive and negative patches from obtained segmenations
        """ 
          
        self.Batches=[]

        count=0
        for i in tqdm(range(np.shape(self.indKeys)[0])):
            idx=self.indKeys[i]
            B=self.checkV(idx)
            
            if B is not None:
                #fix the size.
                inS=np.shape(B)
                B=ndim.zoom(B, (outS[0]/inS[0], outS[1]/inS[1], outS[2]/inS[2]), order=0)
#                B=morph1.remove_small_objects(B, min_size=512)
                self.appendBatch(B)
                count+=1
            
        self.nBatches=count
            

    def procSeg(self):
            
            """
            This funtion perform two stages of morphological processing on the segmente stack. 
            First operation is performed globally and the second one is adaptive.
            """
        
            #first 
            v=morph1.remove_small_holes(self.segData, min_size=8**3)
            v=morph1.remove_small_objects(v, min_size=self.E_param*(32**3))
            v=morph1.binary_closing(v,morph1.ball(3))
            v=morph1.binary_opening(v,morph1.ball(1))
            
            #second (adaptive)
            #v=self.adaptive_erosion(v, alpha=self.AE_param)
            v=morph1.remove_small_holes(v, min_size=8**3)
            v=morph1.remove_small_objects(v, min_size=self.E_param*(32**3))
            self.segData=v

    
    def adaptive_erosion(self, v, alpha):    
        
        """
        This function perform adaptive erosion based on max. distance transform
        """        
        dist_v=getDist(v)  # based on 2D sectioning to cal. max dist        
        s=int(np.max(dist_v))
        max_v=filt.maximum_filter(dist_v,size=(s,s,s)) 
        max_v=np.square(max_v)
        out_v=np.divide(dist_v,(max_v+1))    
        out_v=out_v>alpha
        out_v=morph1.remove_small_holes(out_v, min_size=512)
        out_v=morph1.remove_small_objects(out_v, min_size=self.E_param*512)  
        v=morph1.binary_closing(v,morph1.ball(3))
        out_v=morph1.binary_opening(out_v,morph1.ball(1))
  
        return out_v
        
    def getSkl(self):
        self.sklData=sk3d(self.segData.astype('int')).astype('bool')
        
    def getDist(self):         
        self.distDataF=getDist(self.segData.astype('int'))
        self.distData=self.distDataF*self.sklData
         
    def getKeys(self):     
        sk_=(filt.uniform_filter(self.sklData.astype('float'), size=3)*27).astype('int')
        self.keysData=sk_>3
        
    def getSmooth(self, sigma=1, mode='same'):
        
        d=self.distDataF
        d=filt.gaussian_filter(d,sigma)
        
        s=d*self.sklData
        m=np.zeros_like(s)
        
        indices=np.where(s>0)
        indices=np.array([indices[0],indices[1],indices[2]]).T # from rows in tuple to columns in np array  
        
        for i in tqdm(range(np.shape(indices)[0])):
            
            idx=indices[i]
            x0=idx[1]
            y0=idx[2]
            z0=idx[0]
            
            # radius of the sphere and numbr of point grids
            l=s[z0,x0,y0]  
            n_p=int(2*l+1)
            
            # get the grid for x, y, z
            x=np.floor(np.linspace(x0-l,x0+l,n_p))
            y=np.floor(np.linspace(y0-l,y0+l,n_p))
            z=np.floor(np.linspace(z0-l,z0+l,n_p))
            
            # get the meshgrid and build the sphere
            p1,p2,p3=np.meshgrid(x,y,z);                                
            sphere=((p1-x0)**2+(p2-y0)**2+(p3-z0)**2)<=l**2

            
            p11=p1*sphere;  p11=np.ndarray.flatten(p11)
            p22=p2*sphere;  p22=np.ndarray.flatten(p22) 
            p33=p3*sphere;  p33=np.ndarray.flatten(p33)
       
            # set the grid points of the origional data stack that 
            # intersect with the sphere to value 1   
            p=np.array([p33,p11,p22]).astype('int')
            p=tuple(p)
            m[p]=1
            
        if mode =='same':
            self.segData=m
        else:
            self.smoothedSegData=m
                         
            
                
        
    
    
    
    ###  Second Method  ###

    def cvh(self, v):
        points=np.array(np.where(v>0)).T
        return np.sum(v)/fc(points).volume 

    def getBatches2(self, ind, thresh=[0,1], c=.75):
        #initialize
        batches=[]
        selec_ind=np.zeros(np.shape(ind)[0]*len(self.pSize[0]))

        for l,m,n in zip(self.pSize[0],self.pSize[1],self.pSize[2]):
            print('\n'+'*'*75+'\n'+'Extracting batches at scale: '+'['+str(l)+', '+str(m)+', '+str(n)+']')
            s_z=int(l/2)# step in z       
            s_x=int(m/2)# step in x 
            s_y=int(n/2)# step in y
            for i in tqdm(range(np.shape(ind)[0])):
                
                z,x,y=ind[i]  
                # check if the patch does not exceed the bounds
                if ((self.d1-s_z > z >= s_z+1) and
                    (self.d2-s_x > x >= s_x+1) and 
                    (self.d3-s_y > y >= s_y+1)):      
                    batch=self.segData[z-s_z:z+s_z , x-s_x:x+s_x, y-s_y:y+s_y] 
                    
                    # check if the patch is representative
                    if thresh[0]*m*n*l <= np.sum(batch>0) <=thresh[1]*m*n*l:                        
                        
                        #check if a batch at this index is not added
                        if selec_ind[i]==0:
                            
                            #fix the size if ness.
                            self.wanted_size=s=[16.0,32.0,32.0]
                            if not np.array_equal(s,[l,m,n]):
                               batch=ndim.zoom(batch, (s[0]/l, s[1]/m, s[2]/n), order=0)
 
                            batches.append(batch)# add the patch to the dataset
                            selec_ind[i]=1
                                 
        return selec_ind, np.array(batches[:])
    



class arg():
    def __init__(self, batch=1, nb_classes=2, w_path=None):
        self.batch=batch
        self.nb_classes=nb_classes

# perform DBSCAN clustering fot the output of prediction
def getCluster(X, eps=0.05, min_samples=20):
    
    #find labels
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # Number of clusters in pred labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    def computeCentroid(points):
        x1 = map(lambda p: p[0], points)
        x2 = map(lambda p: p[1], points)
        x3 = map(lambda p: p[2], points)
        centroid = [np.mean(x1), np.mean(x2), np.mean(x3)]
        return centroid

    #exclude noise from pred (outliers)
    ind=np.where(labels >-1)
    X_=np.array(X[ind])
    labels_=np.array(labels[ind])
 
    #calculate centroids
    points=dict()
    centroids=dict()  
    for i in range(n_clusters): 
        ind=np.where(labels_ ==i)
        points[i]=X_[ind]
        centroids[i]=computeCentroid(points[i])
        X=np.vstack((X,np.array(centroids[i])))
        labels=np.append(labels,n_clusters+10)
        
    nodes=[]
    for key, value in centroids.iteritems():
        temp = [key,value]
        nodes.append(temp[1])
    nodes=np.array(nodes)
        
    return nodes


#plot scatter points
def plot(v):
    indices=np.where(v>0)
    x1,x2,x3=indices[0],indices[1],indices[2]   
    d=mlab.pipeline.scalar_scatter(x1,x2,x3)
    g=mlab.pipeline.glyph(d)
#    e=mlab.get_engine()
#    g=e.scenes[0].children[0].children[0].children[0]
#    g.glyph.glyph.range = np.array([ 0.,  1])
#    g.glyph.glyph.scale_factor = 0.1  

def makeMovie(path):

    for i in range(360):
        # Rotate the camera by 10 degrees.
        e=mlab.get_engine()
        c=e.current_scene
        c.scene.camera.azimuth(1)
    
        # Resets the camera clipping plane so everything fits and then
        # renders.
#        c.scene.reset_zoom()

        # Save the scene.
        c.scene.save_png(path+'anim%d.png'%i) 
        #"ffmpeg -f image2 -r 25 -i anim%d.png -vcodec mpeg4 -vb 20M  -y movie.mp4"




def getGraphfromCGAL(filenameVertices, filenameEdges, FullyCC=True):
    
    P1, P2, P11, P22=readCGAL(filenameEdges, filenameVertices)   
    p, intersections, c=getGraph(P1, P2, P11, P22) 
    
    G=nx.Graph()
    G.add_nodes_from(range(np.shape(p)[0]))
    G.add_edges_from(np.ndarray.tolist(c))
    for i in range(np.shape(p)[0]):
        G.node[i]['pos']=p[i,:]
    G.to_undirected()
    
    if FullyCC==True:
        
        # connected components
        graphs=list(nx.connected_component_subgraphs(G))
        s=0
        ind=0
        for idx, i in enumerate(graphs):
            if len(i)>s:
                s=len(i); ind=idx
        G=graphs[ind]
        G=fixG(G) 
        
    return G


def getGraph(P1, P2, P11, P22):

    u1=np.unique(P1,axis=0)
    u2=np.unique(P2,axis=0)
    P_unique=np.unique(np.vstack([u1,u2]),axis=0)
    
    
    labels=dict()
    for i in range(len(P_unique)):
        labels[str(P_unique[i,:])]=i    

    
    counter=dict()
    intersect_ind=list()



    counter1=dict()
    for i in P1:
        if str(i) in counter1.keys():        
            counter1[str(i)]=counter1[str(i)]+1
        else:
            counter1[str(i)]=1   
#    
                 
    counter2=dict()    
    for i in P2:
        if str(i) in counter2.keys():        
            counter2[str(i)]=counter2[str(i)]+1
        else:
            counter2[str(i)]=1  

    
    for i in labels.keys():    
        if i in counter1.keys():
            if counter1[i]>2:
                intersect_ind.append(labels[i])
            else:
                if counter1[i]==2 and i in counter2.keys():
                    intersect_ind.append(labels[i])
        if i in counter2.keys():  
            if counter2[i]>2:
                intersect_ind.append(labels[i])
            else:
                if counter2[i]==2 and i in counter1.keys():
                    intersect_ind.append(labels[i])


    intersect=P_unique[intersect_ind]    

       
    connections=[]
    for i in range(len(P1)):
        start=labels[str(P1[i])]
        end=labels[str(P2[i])]
        connections.append((start,end))
        
    return P_unique, intersect, np.array(connections)



def readCGAL(filenameEdges, filenameVertices):
    f_edges = open(filenameEdges, 'r')
    c_edges=f_edges.readlines()
   
    f_verts = open(filenameVertices, 'r')
    c_verts=f_verts.readlines()
    
    def process(c):
        c=c.rstrip('\n')
        c=c.split()    
        for i in range(len(c)):
            c[i]=float(c[i])        
        return c
            
    c_edges=[process(c_edges[i]) for i in range(len(c_edges))]    
    p_edges=np.array(c_edges)
    P1=p_edges[:,0:3]
    P2=p_edges[:,3:6]
    
    c_verts=[process(c_verts[i]) for i in range(len(c_verts))]    
    p_verts=np.array(c_verts) 
    P11=p_verts[:,0:3]
    P22=p_verts[:,3:6]
              

    return P1, P2, P11, P22

def readOFF(filename):
    """
    Reads vertices and faces from an off file.
 
    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """
 
    assert os.path.exists(filename)
 
    with open(filename, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]
 
        assert lines[0] == 'OFF'
 
        parts = lines[1].split(' ')
        assert len(parts) == 3
 
        num_vertices = int(parts[0])
        assert num_vertices > 0
 
        num_faces = int(parts[1])
        assert num_faces > 0
        
        if lines[2]=='':                
            K=3
        else:
            K=2
                    
        vertices = []
        for i in range(num_vertices):
            try:
                vertex = lines[K + i].split(' ')
                vertex = [float(point) for point in vertex]
                assert len(vertex) == 3     
                vertices.append(vertex)
            except ValueError,e:            
                 print "error",e,"on line",i
 
        faces = []
        for i in range(num_faces):
            try:                
                face = lines[K + num_vertices + i].split()
                face = [int(index) for index in face if index !='']
            except ValueError,e:
                print "error",e, "on line",i, face[i]
    
            assert face[0] == len(face) - 1
            for index in face:
                assert index >= 0 and index < num_vertices
 
            assert len(face) > 1
 
            faces.append(face[1:4])
 
        return [vertices, faces]

       
def visGraph(p, e, d, radius=.15, color=(0.8,0.8,0), gylph_r=0.5, gylph_c=(1,1,1)):
    
    
    if d:
        pts=mlab.points3d(p[:,0],p[:,1],p[:,2], d)
    else:
        pts=mlab.points3d(p[:,0],p[:,1],p[:,2], scale_factor=gylph_r, color=gylph_c)

    pts.mlab_source.dataset.lines = e 
    tube = mlab.pipeline.tube(pts, tube_radius=radius)
    tube.filter.set_input_data(pts.mlab_source.dataset)    
    surface=mlab.pipeline.surface(tube, color=color)

    if d:
        tube.filter.vary_radius = 'vary_radius_by_absolute_scalar'
        surface.actor.mapper.scalar_range = [ 0.0, np.max(d)]
        surface.actor.mapper.progress = 1.0
        surface.actor.mapper.scalar_visibility = True

def visG(G, 
         radius=.15, 
         color=(0.8,0.8,0), 
         gylph_r=0.5, 
         gylph_c=(1,1,1),
         diam=False,
         jnodes_r=None,
         jnodes_c=(0,0,1)):
    
    nodes=[G.node[i]['pos'] for i in G.nodes()]   
    edges=[[i[0],i[1]] for i in G.edges()]    
    
    #obtain radius in each node
    d=None
    if diam:
        try:
            d=[G.node[i]['d'] for i in G.nodes()]   
        except:
            pass
    
    nodes=np.array(nodes).astype('float') # importnat to be float type
    edges=np.array(edges)
    
    visGraph(p=nodes, e=edges, d=d,
             radius=radius,
             color=color, 
             gylph_r=gylph_r, 
             gylph_c=gylph_c)
    
    if jnodes_r:
        _, jnodes=findNodes(G, j_only=False)  
        x=[i[0] for i in jnodes]
        y=[i[1] for i in jnodes]
        z=[i[2] for i in jnodes]
        mlab.points3d(x,y,z,scale_factor=jnodes_r, color=jnodes_c)




        
        
def getBranches_no(intersections):    
    
    branches=[]
    points=[]
    Input=intersections
    px=np.array([np.inf,np.inf,np.inf])
    
    for i in range(np.shape(intersections)[0]):
        
        while px not in points:
            
            Input=intersections
            p1=intersections[i,:] # select a point
            points.append(p1)
            
            # delet it from Input
            ind_p1=np.where(np.all(Input==p1, axis=1))
            Input=np.delete(Input,ind_p1,axis=0)
            p1.shape=(1,3)
            
            # compute ecludien distance of Input from p1
            p1_stack=np.repeat(p1, np.shape(Input)[0], axis=0)
            dist=np.sqrt(np.sum(np.square(Input-p1_stack), axis=1))
            
            # find the ind of point (px) in Input that has the min dist with p1
            ind=np.where(dist==np.min(dist))
            ind=np.array(ind).ravel() 
            ind=ind[0]
            px=Input[ind]
            
            points.append(px)
        
        branches.append([p1.ravel(),px])
        
    return branches
    
    
def fixG(G):
    Oldnodes=G.nodes()
    new=range(len(Oldnodes))
    mapping={Oldnodes[i]:new[i] for i in new}
    G=nx.relabel_nodes(G, mapping)
    return G
    
def removeEndsG(G):
    NodesToRemove=[0]
    while len(NodesToRemove)!=0:
        NodesToRemove=[]
        for i in G.nodes_iter():
            k=nx.neighbors(G, i)
            if len(k)==1:
                NodesToRemove.append(i)    
        G.remove_nodes_from(NodesToRemove)
    return G
            
def getGraphfromMat(filename):
    
    F=h5py.File(filename, 'r')
    In=F['im2']
    items=In.keys()
    data=dict()
    for i in items:   
        data[i]=np.array(In[i])
    

    nX=data['nX']
    nY=data['nY']
    nZ=data['nZ']
    
    scale=data['Hvox']

        
    xx=int(nX*scale[0])
    yy=int(nY*scale[1])
    zz=int(nZ*scale[2])
       
    # read nodes 
    Pos=data['nodePos_um'].T
    
    # read edges
    Edg=(data['nodeEdges'].T).astype('int')
    connections=[]
    for i in range(len(Edg)):
        connections.append((Edg[i,0]-1,Edg[i,1]-1))
    
    # graphing funtion
    x,y,z=Pos[:,0].astype('float'), Pos[:,1].astype('float'), Pos[:,2].astype('float')
    
    G=nx.Graph()
    G.add_nodes_from(range(np.shape(Pos)[0]))
    G.add_edges_from(connections)
    
    x=256
    y=256
    
    minx=np.min(Pos[:,0])
    miny=np.min(Pos[:,1])
    minz=np.min(Pos[:,2])
    
    maxx=np.max(Pos[:,0])
    maxy=np.max(Pos[:,1])
    maxz=np.max(Pos[:,2])
        
    for i in range(np.shape(Pos)[0]):
        
        px=512*((Pos[i,0]-minx)/maxx)
        
        py=512*((Pos[i,1]-miny)/maxy)
        
        pz=512*((Pos[i,2]-minz)/maxz)
        
        G.node[i]['pos']=np.array([py,y-(px-y),pz])
        
    return G


def createCam(position,
             focal_point,
             view_angle,
             view_up,
             clipping_range):
    
    cam=dict()
    
    cam['position']=position
    cam['focal_point']=focal_point    
    cam['view_angle']=view_angle
    cam['view_up']= view_up   
    cam['clipping_range']=clipping_range
    
    return cam
    


def setCam(cam=None):
    
    if cam:  
        
        e=mlab.get_engine()
        c=e.current_scene
        c.scene.camera.position = cam['position']
        c.scene.camera.focal_point = cam['focal_point']
        c.scene.camera.view_angle = cam['view_angle']
        c.scene.camera.view_up = cam['view_up']
        c.scene.camera.clipping_range = cam['clipping_range']
        c.scene.camera.compute_view_plane_normal()
        c.scene.render()    
    
    
def getSlicesfromGraph(G, name='graphSlices', mode='slices', image=False): 
   
    fig=mlab.figure(bgcolor=(0,0,0),size=(512,561))        
    
    def setCamera():
        e=mlab.get_engine()
        c=e.current_scene
        c.scene.camera.azimuth(1)
        c.scene.camera.position = [255.5, 255.5, -1125]
        c.scene.camera.focal_point =  [255.5, 255.5, 255.0]
        c.scene.camera.view_angle = 30
        c.scene.camera.view_up =  [1, 0, 0]
        c.scene.camera.clipping_range = [0, 3000]
        c.scene.camera.compute_view_plane_normal()
        c.scene.render() 
        depth=c.scene.camera.position
        return c, depth[2]
        
    visG(G, radius=2, color=(0,1,0))
    
    if image:
        scr=mlab.pipeline.scalar_field(stk)
        imageCut=mlab.pipeline.scalar_cut_plane(scr)
        imageCut.actor.property.opacity=.2
        imageCut.implicit_plane.widget.normal_to_z_axis=True
        
        
    e=mlab.get_engine()
    gylph= e.scenes[0].children[0].children[0].children[0]
    gylph.actor.actor.visibility=False
    surface = e.scenes[0].children[0].children[1].children[0].children[0]
    surface.actor.actor.visibility=False
    
    cut_plane = mlab.pipeline.scalar_cut_plane(surface,
                                plane_orientation='z_axes')
    cut_plane.implicit_plane.origin = (256, 256, 0)
    cut_plane.actor.property.color = (1, 1, 1)
    cut_plane.implicit_plane.widget.enabled = False   
    c,depth=setCamera()

    slices=[]
    for i in range(512):
        cut_plane.implicit_plane.origin = (256, 256, i)
        c.scene.camera.position = [255.5, 255.5, depth+i]
        if image: 
            imageCut.implicit_plane.origin = (256, 256, i)
        
        s = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
        s=ndim.filters.maximum_filter(s[:,:,0]>0,size=(3,3))
        #plt.imshow(s)
        im=Image.fromarray(np.array(s*255, dtype=np.uint8))        
        if mode=='slices':
            im.save(name+str(i)+'.png')
        else:
            slices.append(np.array(s*255, dtype=np.uint8))
            
    if len(slices)>0:
        slices=np.array(slices)
        slices=slices[:,:,:]
        io.imsave(name,slices) 
        
def displayAdj(G):        
    admG=nx.adjacency_matrix(G).toarray()
    scatterG=np.where(admG==1)
    x=scatterG[0]
    y=scatterG[1]
    plt.figure()
    plt.scatter(x,y)
    
def reduceG(G, j_only=False):
    cont=1
    idNodes,_=findNodes(G, j_only=j_only)
    
    while cont!=0:
    #        NodesToRemove=[]
    #        NodesToConnect=[]
        cont=0
        for i in list(G.nodes_iter()):
            k=nx.neighbors(G, i)
            if len(k)==2:
                if i not in idNodes:
                    G.remove_node(i)
                    G.add_edge(k[0],k[1])
                    cont=1
                
    #to account for loopy structure
    
    
    #G=fixG(G)
    return G

def visPoints(points, scale_factor=1, color=(1,1,1)):
    x=[i[0] for i in points]
    y=[i[1] for i in points]
    z=[i[2] for i in points]
    mlab.points3d(x, y, z, 
                  scale_factor=scale_factor,
                  color=color)
    e=mlab.get_engine()

def findNodes(G, j_only=False):
    
    nodes=[]
    ind=[]
    
    if j_only:      
        
        for i in G.nodes():
            if len(G.neighbors(i))==3:
                nodes.append(G.node[i]['pos'])
                ind.append(i)      
    else:
               
        for i in G.nodes():
            if len(G.neighbors(i))!=2:
                nodes.append(G.node[i]['pos'])
                ind.append(i)
                
    try:       
        nodes=[i.tolist() for i in nodes]
    except:
        pass

    return ind, nodes
   
def findPoints(G):
    p=[]
    for i in range(G.number_of_nodes()):        
        p.append(G.node[i]['pos'].tolist())         
    return p

def findEnds(G):
    p=[]
    for i in range(G.number_of_nodes()):
       if len(G.neighbors(i))==1:
           p.append(i)
    return p
  

    
def prunG(G):

    """
    This function remove branches of length =1.
    
    Input:
        "G": NetworkX undirected graph
        
    Output:
        "G": pruned version of the intput "G"
        
    """
    j_nodes,_=findNodes(G)  
    
    nbrs=[]
    for i in j_nodes:
        nbrs.append(nx.neighbors(G,i))
        
    for n, nb in zip(j_nodes, nbrs):
        if len(nb)==1:
            if nb[0] in j_nodes:
                G.remove_node(n)
    G=fixG(G)
    
    return G
    
# This does not work !!!!!!!
#def prunG(G):
#    endP=findEnds(G)
#    crossP,_=findNodes(G)
#    nodes_to_remove=[]
#    for i in crossP:
#        n=G.neighbors(i)
#        if len(n)>2:
#            for j in n:
#                if j in endP:
#                    nodes_to_remove.append(j)
#    G.remove_nodes_from(nodes_to_remove)
#    return fixG(G)

def rescaleG(G, cube=512):
    
    p=findPoints(G)
    pmin=np.min(p,axis=0)  
    pmax=np.max(p,axis=0) 
    
    for i in range(G.number_of_nodes()):
        p_=G.node[i]['pos']
        G.node[i]['pos']=512*(p_-pmin)/pmax
        
    return G
 
def saveSceneToPNG(name):
    #save as PNG
    s = mlab.screenshot( mode='rgba', antialiased=True)
    im=Image.fromarray(np.array(s*255, dtype=np.uint8))
    im.save(name)
    

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

def getCoreGraph(G, indices):
    
    #obtain reduced graphs (with only burifications)
    Gr=reduceG(G.copy()) # note that Gr is not fixed by fixG
    
    nodesGr=Gr.nodes()
    
    nbrs=[]
    #find connections between nodes
    #connections=[]
    for i in Gr.nodes():
        nbrs_=Gr.neighbors(i)
        nbrs.append(nbrs_)
        #connections_=[[i,j] for j in nbrs_]
        #connections.append(connections_)
    
    #find the complete branches(pathes) between nodes
    pathes=[]   
    for i, j in zip(nodesGr, nbrs):
        pathes_=[]
        for nbr in j: 
            pth=nx.shortest_path(G, i, nbr)
            pth.pop() # to avoid removing the node on the other side of the branch
            pathes_.append(pth)
        pathes.append(pathes_)
    
    pathes_to_remove=[]
    for id_i, i in enumerate(nodesGr):
        if i in indices:
            pathes_to_remove.append(pathes[id_i])
    
    #find nodes and points on the corresponding branches to be removed 
    nodes_to_remove=list(set([i[j][k] for i in pathes_to_remove 
                     for j in range(len(i))
                     for k in range(len(i[j]))]))
    
    for i in nodes_to_remove:
        G.remove_node(i)
        
    for i in G.nodes():
        if len(G.neighbors(i))<1:
            G.remove_node(i)
    
    return G


def getBranches(G):
    
    idNodes,_=findNodes(G)
    Gr=reduceG(G.copy())
    nbrs=[]    
    
    #find connections between nodes    
    for i in idNodes:
        nbrs_=Gr.neighbors(i)
        nbrs.append(nbrs_)
    
    #find the complete branches(pathes) between nodes
    pathes=[]  
    for i, j in zip(idNodes, nbrs):
        for nbr in j: 
            pth=set(nx.shortest_path(G, i, nbr))
            if not pth in pathes:
                pathes.append(pth) 
    pathes=[list(i) for i in pathes]
    return pathes



# get graph in the middle
def  getMiddleGraph(G, thr):

    max_p1=0
    max_p2=0
    max_p3=0
    
    for i in G.nodes():
        
        if max_p1<G.node[i]['pos'][0]:
            max_p1=G.node[i]['pos'][0]
        
        if max_p2<G.node[i]['pos'][1]:
            max_p2=G.node[i]['pos'][1]
        
        if max_p3<G.node[i]['pos'][2]:
            max_p3=G.node[i]['pos'][2]

    
    
    for i in G.nodes():
        p1,p2,p3=G.node[i]['pos']       
        if p1>thr and p1<max_p1-thr and p2>thr and p2<max_p2-thr and p3>thr and p3<max_p3-thr:
            pass
        else:
            G.remove_node(i)

    return fixG(G)
        
        
def getFullyConnected(G):
    
    
    # connected components
    graphs=list(nx.connected_component_subgraphs(G))
    s=0
    ind=0
    for idx, i in enumerate(graphs):
        if len(i)>s:
            s=len(i); ind=idx
    G=graphs[ind]
    G=fixG(G) 
      
    return G


import SimpleITK as sitk

def makeMIP(array=None, image_path=None, output_dir=None, output=True):
    
    if array is not None:
        image=sitk.GetImageFromArray(array)
        
    if image_path is not None:
        image = sitk.ReadImage(image_path)   
        basename = os.path.basename(image_path)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            
    image_size = image.GetSize()
    for dim in range(3):
        projection = sitk.MaximumProjection(image, dim)

        if image_size[dim] % 2:  # odd number
            voxel = [0, 0, 0]
            voxel[dim] = (image_size[dim] - 1) / 2
            origin = image.TransformIndexToPhysicalPoint(voxel)
        else:  # even
            voxel1 = np.array([0, 0, 0], int)
            voxel2 = np.array([0, 0, 0], int)
            voxel1[dim] = image_size[dim] / 2 - 1
            voxel2[dim] = image_size[dim] / 2
            point1 = np.array(image.TransformIndexToPhysicalPoint(voxel1.tolist()))
            point2 = np.array(image.TransformIndexToPhysicalPoint(voxel2.tolist()))
            origin = np.mean(np.vstack((point1, point2)), 0)
        projection.SetOrigin(origin)
        projection.SetDirection(image.GetDirection())
     
        if output_dir:
            proj_basename = basename.replace('.nii.gz', '_mip_{}.nii.gz'.format(dim))
            sitk.WriteImage(projection, os.path.join(output_dir, proj_basename))
        
    if output:
        return sitk.GetArrayFromImage(projection)


     
        
def visProG(G, radius=.15, 
            color=(0.8,0.8,0), 
            gylph_r=1, 
            gylph_c=(1,1,1), 
            bound=1, 
            bgcolor=(1,1,1), 
            init=False):
    
    
#        figure = mlab.gcf()
#        mlab.clf()
    figure=mlab.figure(bgcolor=bgcolor)
    figure.scene.disable_render = True
    
    def showG(P_unique, connections, radius=.15, color=(0.8,0.8,0), gylph_r=0.5, gylph_c=(1,1,1)):
    
        pts=mlab.points3d(P_unique[:,0],P_unique[:,1],P_unique[:,2],scale_factor=gylph_r, color=gylph_c)
        pts.mlab_source.dataset.lines = np.array(connections)
        
        tube = mlab.pipeline.tube(pts, tube_radius=radius)
        tube.filter.radius_factor = 1.
        tube.filter.vary_radius = 'vary_radius_by_scalar'
        tube.filter.set_input_data(pts.mlab_source.dataset)
        mlab.pipeline.surface(tube, color=color)
        
        return pts
        
    
    def inG(G, radius=.15, color=(0.8,0.8,0), gylph_r=0.5, gylph_c=(1,1,1)):
        nodes=[]
        for i in range(G.number_of_nodes()):
            nodes.append(G.node[i]['pos'])
        edges=[]
        for i in G.edges_iter():
            e=[i[0],i[1]]
            edges.append(e)
            
            
        nodes=np.array(nodes)
        edges=np.array(edges)
        
        pts=showG(nodes, edges, 
                 radius=radius, 
                 color=color, 
                 gylph_r=gylph_r, 
                 gylph_c=gylph_c)
        
        return pts
    
    g_points=np.array(findPoints(G))        
    
    if init:
        p1,p2,p3=[init[0], init[1], init[2]]
    else:
        p1,p2,p3=g_points[1]


    pts=inG(G, 
         radius=radius, 
         color=color, 
         gylph_r=gylph_r, 
         gylph_c=gylph_c)
    
#            outline = mlab.outline(line_width=5)
#            outline.outline_mode = 'full'
#            outline.bounds = (p1-bound, p1+bound,
#                              p2-bound, p2+bound,
#                              p3-bound, p3+bound)
    
    sphere = mlab.points3d(p1, p2, p3, scale_mode='none',
                    scale_factor=bound,
                    color=(1, 1, 1),
                    opacity=.7)
    
    figure.scene.disable_render = False
    
    glyph_points = pts.glyph.glyph_source.glyph_source.output.points.to_array()
    
    
    def picker_callback(picker):
        """ Picker callback: this get called when on pick events.
        """
        if picker.actor in pts.actor.actors:

            point_id = picker.point_id/glyph_points.shape[0]
            # If the no points have been selected, we have '-1'
            if point_id != -1:

                x, y, z = g_points[point_id]
                
#                        # Move the outline to the data point.
#                        outline.bounds = (x-bound, x+bound,
#                                          y-bound, y+bound,
#                                          z-bound, z+bound) 
                                       
                sphere.mlab_source.set(x=x,y=y,z=z)
                
                print point_id
                print x,y,z
    
    picker = figure.on_mouse_pick(picker_callback)
    picker.tolerance = 0.001



def getPropagate2(G, idx, cutoff=10):
        
    path=nx.single_source_shortest_path(G, idx, cutoff=cutoff)
    end_points=[i for i in path.keys()]
    path_p=[path[i] for i in path.keys()]
    
    path_points=[]
    
    for i in path_p:
       
        try:
            len(i)
            for j in i:
                path_points.append(j)
        except:
                path_points.append(i)    
     
    points= set(path_points).union(set(end_points))
    points=list(points)        
    
    for i in G.nodes():
        if i in points:
            pass
        else:
            G.remove_node(i)
            
    return fixG(G) 


def getPropagate(G, idp, cutoff=5):
        
    Gc=reduceG(G.copy())
    
    #get other juction nodes through the pathes 
    J=nx.single_source_shortest_path(Gc, idp, cutoff=cutoff)
    eJ=[i for i in J.keys()]
    pJ=[J[i] for i in J.keys()]
    
    pairJ=[]
    for i in pJ:
        l=len(i)
        if len(i)>1:
            for idx, j in enumerate(i):
                if idx<(l-1):
                    pairJ.append([j,i[idx+1]])
    
    #all nodes in the pathes between j nodes
    pnts=[]
    for i in pairJ:
        pnts.append(nx.shortest_path(G,i[0],i[1]))  
    pnts=[i for j in pnts for i in j]  
    pnts=list(set(pnts))      
  
    
    for i in G.nodes():
        if i in pnts:
            pass
        else:
            G.remove_node(i)
            
    return fixG(G) 

def readPAJEK(filename):
    """
    Reads a Pajek file storing a Graph 
    following the format used in this mudule.
    
    Input:
        "filename": The full directory to Graph file.
    """
    
    G_seg=nx.read_pajek(filename)    
    
    
    G=nx.Graph() 
    
    # build geometry
    for i in range(G_seg.number_of_nodes()):
        node=G_seg.node[str(i)]      
        
        # add node
        n=int(node['id'].encode())
        G.add_node(n-1)
        
        #add position
        pos=node['pos'].encode()
        pos=pos.split(' ')
        
        xyz=[]
        for j in range(len(pos)):         
            try:                
                value=float(pos[j])
                xyz.append(value)
            except:
                try:
                    value=pos[j].split('[')
                    xyz.append(float(value[1]))
                except:
                    try:
                        value=pos[j].split(']')
                        xyz.append(float(value[0]))
                    except : pass
        G.node[i]['pos']=np.array(xyz)
        
        # add label
        try:
            yORn=node['node'].encode()
            if yORn=='False':
                G.node[i]['node']=False
            else:
                G.node[i]['node']=True
        except:
            pass
                     
    #build Topology
    connections_=G_seg.edges()   
    connections=[[int((connections_[i][0]).encode()), int((connections_[i][1]).encode())] for i in range(len(connections_))]
    G.add_edges_from(connections)
        
    return G


def fixConnections(G):
    # find wrong nodes and their positions        
    
    w_nodes=[] # wrong nodes
    w_p=[] # position of wrong nodes
    
    w_nbrs=[]  # wrong neighbors      
    w_pos=[] # position of wrong neigbors
    
    for i in G.nodes():
        w_nbrs_=G.neighbors(i)
        if len(w_nbrs_)==4:
            w_nodes.append(i)
            w_p.append(G.node[i]['pos'])
            w_nbrs.append(w_nbrs_)
            w_pos_=[G.node[j]['pos'] for j in w_nbrs_]
            w_pos.append(np.array(w_pos_))
  
    # decide which nodes to be connected avoiding the juction
    
    ind_nodes=[] # indcies of nodes to avoid junction
    a=[]
    ind=[]
    kk=list(it.combinations([0,1,2,3],2))       
    for i, j in zip(w_p, w_pos):       
        a_=[]

        for k in kk:
            p=[i, j[k[0]], j[k[1]]]           
            a_.append(cycleArea(p))            
        idx=np.argsort(a_)
        ind.append(kk[idx[0]])            
        a.append(a_)

    for i, j in zip(w_nbrs, ind):            
        ind_nodes.append([ i[j[0]], i[j[1]] ])
    
    # modify_graph
    for i, j in zip(w_nodes, ind_nodes):           
        G.remove_edge(i,j[0])
        G.remove_edge(i,j[1])
        G.add_edge(j[0],j[1])
        
    return G



def TrianArea(p1, p2, p3):
    
    def cal_abc(p1,p2,p3):
        a=np.linalg.norm(p1-p2)
        b=np.linalg.norm(p1-p3)
        c=np.linalg.norm(p2-p3)
        return a, b, c
    
    try:
        cal_abc(p1,p2,p3)
    except:
        p1=np.cross
        cal_abc(p1,p2,p3)
    # calculate the sides
    s = (a + b + c) / 2
    # calculate the area
    area = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    return area

def cycleArea(corners):
    n = len(corners) # of corners
    cross= [0.0,0.0,0.0]
    for i in range(n):
        j = (i + 1) % n
        crss=np.cross(corners[i],corners[j])
        cross=cross+crss
        #nrm+=np.linalg.norm(crss)
    area = np.linalg.norm(cross) / 2.0
    return area

def eigV(p):
    
    centered_matrix = p - p.mean(axis=1)[:, np.newaxis]
    cov = np.dot(centered_matrix, centered_matrix.T)
    eigvals, _ = np.linalg.eig(cov)
    delta=np.exp(1/(eigvals[0]+eigvals[1]))-1
    return delta
    
def angle(a, p1, p2):
  
    ap1 = p1 - a
    ap2 = p2 - a
    
    x1,x2,y1,y2,z1,z2=ap1[0],ap2[0],ap1[1],ap2[1],ap1[2],ap2[2]    
    dot=x1*x2+y1*y2+z1*z2
    norm1=(x1**2+y1**2+z1**2)**.5
    norm2=(x2**2+y2**2+z2**2)**.5 
    
    if norm1>0 and norm2>0:
        cosine_angle =  dot/(norm1*norm2)  
    else:
        cosine_angle =  1
    
    add=0
   
    cosine_angle=round(cosine_angle,5) 
    
    if cosine_angle <0:
        add=np.pi/2
        cosine_angle+=1        
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle+add)


def isSklNode(node, p, thr):
   
    thr1, thr2=thr,180-thr
    
    idx=range(np.shape(p)[0])    
    idx.pop(0)
    
    skl_node=True
    
    for i in idx:    
        
        ang=angle(node, p[0], p[i])  
       
        if ang>thr1 and ang<thr2:
            skl_node=False
            
    return skl_node

def getPropagate_branches(G, idp, cutoff=5):
    
    
    Gc=reduceG(G.copy(), j_only=True)
    
    #get other juction nodes through the pathes 
    J=nx.single_source_shortest_path(Gc, idp, cutoff=cutoff)
    J=[J[i] for i in J.keys()]
    
    
    # get branches at different levels
    levels=dict()
    for j in range(cutoff):
        branches_=[]
        for i in J:
            if len(i)>1:
                try:
                    branches_.append([i[j],i[j+1]]) 
                except:
                    pass
        
        branches_=[tuple(i) for i in branches_]
        branches_=list(set(branches_))
        branches_=[list(i) for i in branches_]
        levels[j]=(branches_)
    
    
    #all nodes in the pathes between pair of j nodes
    b=dict()
    
    for i in levels.keys():
        pairJ=levels[i]
        pnts=[]
        for j in pairJ:
            pnts.append(nx.shortest_path(G,j[0],j[1]))  
        b[i]=pnts            
        
       
  
  
    Graphs=dict()#  represent each level as a graph
    
    for k in b.keys():
        
        pnts=b[k]
        pnts=[i for j in pnts for i in j]  
        G_=G.copy()
        
        for i in G_.nodes():
            if i in pnts:
                pass
            else:
                G_.remove_node(i)
        G_=fixG(G_)       
        Graphs[k]=G_
        
         
    return Graphs  

def visGs(Graphs, radius=.5, bound=5, gylph_r=2):
    
    colors=[(0.0, 0.0, 0.0),
             (0.0, 0.0, 1.0),
             (0.6470588235294118, 0.16470588235294117, 0.16470588235294117),
             (0.8705882352941177, 0.7215686274509804, 0.5294117647058824),
             (0.37254901960784315, 0.6196078431372549, 0.6274509803921569),
             (1.0, 0.3803921568627451, 0.011764705882352941),
             (0.23921568627450981, 0.34901960784313724, 0.6705882352941176),
             (0.5019607843137255, 0.5411764705882353, 0.5294117647058824),
             (1.0, 0.5490196078431373, 0.0),
             (0.0, 0.7490196078431373, 1.0)]
    
    for i in Graphs.keys():
        if i <1:
            visProG(Graphs[i], radius=radius, color=colors[i], bound=7, gylph_r=gylph_r)
        else:
            visG(Graphs[i], radius=radius, color=colors[i], gylph_r=gylph_r)


def visMesh(m, color=(1,0,0), opacity=.5, 
            figure=True, 
            axes=False,
            adjust=False,
            wire=True):
    
    #build pipeline
    
    if figure:
        mlab.figure(bgcolor=(1,1,1))
        
        
    x,y,z=[i[0] for i in m[0]], [i[1] for i in m[0]], [i[2] for i in m[0]]
    
    if adjust:
        # the next line is to fix the coordinates in the meshes datasets
        y=max(y)-np.array(y)
        y=y.tolist()
    
    t=[i for i in m[1]]
    if adjust:
        mesh=mlab.pipeline.triangular_mesh_source(y,x,z,t) # coordinates are flipped to error in my meshes
    else:
        mesh=mlab.pipeline.triangular_mesh_source(x,y,z,t) # coordinates are flipped to error in my meshes
    mesh.mlab_source.scalars=[1]*len(t)
    
    s=mlab.pipeline.surface(mesh)
    
    #modify surface
    #s=mlab.get_engine().scenes[0].children[0].children[0].children[0]
    s.module_manager.scalar_lut_manager.lut_mode=u'autumn' # colormap
    
    if opacity:        
        s.actor.property.opacity = opacity

    if color:
        s.actor.mapper.scalar_visibility=False
        s.actor.property.color = color #color
    
    if axes:
        ax=mlab.axes(color=(0,0,1), line_width=2, ranges=[0,512,0,512,0,660])
    
    if wire:
        s.actor.property.representation = 'wireframe'


def adjustMesh(m, flip=[0,0,0], switch=[0, 0, 0], reverse=[0,0,0]):
    '''
    modify mesh coordinates
    
    Input:
        
        m: the input mesh
        
        flip: parameters used to flip or not the coordinates: [x, y, z].
        
        switch: parameters used to switch or not coordinates: [xy, xz, yz]
    '''
    
    v=m[0]
    t=m[1]
    
    x, y, z = [i[0] for i in v], [i[1] for i in v], [i[2] for i in v]
    
    if flip[0]==1:
        x=max(x)-np.array(x)
        x=x.tolist()
        
    if flip[1]==1:
        y=max(y)-np.array(y)
        y=y.tolist()

    if flip[2]==1:
        z=max(z)-np.array(z)
        z=z.tolist()   
    
    if reverse[0]==1:
        x=x[::-1]
      
    if reverse[1]==1:
        y=y[::-1]


    if reverse[2]==1:
        z=z[::-1]
           

    if switch[0]==1:
        h=x
        x=y
        y=h
        
    if switch[1]==1:
        h=x
        x=z
        z=h
        
    if switch[2]==1:
        h=y
        y=z
        z=h 
    
    # rebuild v
    v=[[i, j, k] for i, j, k in zip(x, y, z)]
    
    return [v,t]



def sliceG(G, slc=[0,100], flat=True):
    
    g=G.copy()
    
    p=findPoints(g)
    p=np.array(p)
    x, y ,z = p[:,0], p[:,1], p[:,2]
    
    index=np.ravel(np.where(
            np.logical_or((z<slc[0]),(z>slc[1]))
            ))
    
    g.remove_nodes_from(index)        
    g=fixG(g)
    
    if flat:
        for i in g.nodes():
            p_=g.node[i]['pos']
            g.node[i]['pos']=p_*[1,1,0]
            
    return g  



def adjustGraph(g, flip=[0,0,0], switch=[0,0,0]):
    '''
    modify graph coordinates
    
    Input:
        
        m: the input mesh
        
        flip: parameters used to flip or not the coordinates: [x, y, z].
        
        switch: parameters used to switch or not coordinates: [xy, xz, yz]
    '''
    
    v=[g.node[i]['pos'] for i in g.nodes()]
    
    x, y, z = [i[0] for i in v], [i[1] for i in v], [i[2] for i in v]
    
    if flip[0]==1:
        x=max(x)-np.array(x)
        x=x.tolist()
        
    if flip[1]==1:
        y=max(y)-np.array(y)
        y=y.tolist()
    
    if flip[2]==1:
        z=max(z)-np.array(z)
        z=z.tolist()            
    
    if switch[0]==1:
        h=x
        x=y
        y=h
        
    if switch[1]==1:
        h=x
        x=z
        z=h
        
    if switch[2]==1:
        h=y
        y=z
        z=h 
    gg=g.copy()
    # rebuild v
    for idx, i in enumerate(g.nodes()):
        
        gg.node[i]['pos']=np.array([x[idx], y[idx], z[idx]])
    
    return gg


def visImage(image, opacity=1):
    mlab.figure(bgcolor=(1,1,1))
    im=mlab.imshow(image, 
                extent=[0,512,0,512,0,0], 
                opacity=.5)
    im.map_scalars_to_color=True
    im.actor.opacity=opacity
    im.module_manager.scalar_lut_manager.lut_mode='Reds'



def rescaleM(m):
    
    v=m[0]
    t=m[1]
    
    vmin=np.min(v,axis=0)  
    vmax=np.max(v,axis=0) 
    
    v=512*(v-vmin)/vmax
        
    return [v,t]
    
#    def sliceM(m, slc=[0,100]):
#    
#        x,y,z=[i[0] for i in m[0]], [i[1] for i in m[0]], [i[2] for i in m[0]]
#        
#        z=np.array(z)
#        cond=np.logical_and(z>slc[0], z<slc[1])
#        index=np.ravel(np.where(cond))
#        
#        v=m[0]
#        c=m[1]
#        
#        # obtin sliced vertices
#        v_after=[v[i] for i in index]
#        
#        c=np.array(c)
#        c1, c2, c3 = c[:,0], c[:,1], c[:,2]
#       
#        c_after=[i for i in c]     
#        c_after=c==index[0]
#               
#        return [v_after, c_after]
  




class tree:
    
    '''
    This class is intended to read tree graphs generated by the method in:
        ------------------
        "VascuSynth: simulating vascular trees for generating volumetric image 
        data with ground-truth segmentation and tree analysis"
        ------------------
    
    The class is used to read these trees and transform them into a NetworkX undirected graphs.
    '''
    
    def __init__(self, filename=None, t=None):
        
        if t:
            self.t=t
        else:
            if filename:
                self.filename=filename
                self.readT()
            else:
                raise ValueError('A tree graph is required as input.')
        
    def readT(self):
        
        try:
            self.t=sc.io.loadmat(self.filename)['node'][0]
        except:
            raise ValueError('Cannot read tree graph.')
    
        self.number_of_nodes=len(self.t)
        
        # obtain nodes pos
        self.nodes= [self.t[i]['coord'][0] for i in range(self.number_of_nodes)] 
        self.nodes=np.array(self.nodes)
        
        
        # nodes type
        self.type=[str(self.t[i]['type'][0]) for i in range(self.number_of_nodes)] 
        
        # node parents
        self. parent=[self.t[i]['parent'] for i in range(self.number_of_nodes)]
        self.parent={idx:i[0]-1 for idx, i in enumerate(self.parent) if i.size>0 }
        
        #node children
        self.children=[self.t[i]['children'] for i in range(self.number_of_nodes)]
        self.children={idx:i[0]-1 for idx, i in enumerate(self.children) if i.size>0 }
        
        #parent radius
        self.parent_radius=[self.t[i]['parent_radius'] for i in range(self.number_of_nodes)]    
    
    def getG(self):
       
        '''
        Transform into NetworkX graph
        '''
        
        self.connections=[[i, j] for i in self.children for j in self.children[i]]
        self.G=nx.Graph()
        
        self.G.add_nodes_from(range(self.number_of_nodes))   
        for i, j in zip(self.G.nodes(), self.nodes):
            self.G.node[i]['pos']=j
            
        self.G.add_edges_from(self.connections)
                    
        return self.G
        


def getDisConnGraphs(g):
    
    graphs=[]
    gg=nx.connected_component_subgraphs(g)
    
    loop=1
    while loop==1:
        try:
            graphs.append(fixG(gg.next()))
        except:
            loop=0
            
    return graphs





            
if __name__=='__main__':
    
    data_path='dataset.h5'
    t1=time.time() 
    X, Y = get_dataset(data_path, scale=False)
    print('Time to extract dataset: '+str(time.time()-t1))

    
    #zooming operation
#    p1=X2[100]

#    
#    vis_patch(p1) 
#    e=mlab.get_engine()
#    e.new_scene()
#    vis_patch(p2)



def numpy_fill(data, lens, s=False):
    '''
    Pad an array with different row sizes
    Input:
        data: object array
        lens: length of each row of data
        s=length of each element in the row
    '''
    lens=np.array(lens)
    
    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    if s: 
        out = np.zeros((mask.shape[0],mask.shape[1], s), dtype=data.dtype)
        out[mask, :] = np.concatenate(data)

    else:
        out = np.zeros((mask.shape[0],mask.shape[1]), dtype=data.dtype)
        out[mask] = np.concatenate(data)

    return out.astype(float), mask   


