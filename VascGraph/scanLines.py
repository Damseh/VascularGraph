#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:35:20 2019

@author: rdamseh
"""
# visualization
import matplotlib
import matplotlib.pyplot as plt

# computation
import numpy as np
import scipy as sp
import sklearn as sk
import networkx as nx
import scipy.ndimage.morphology as morph
import scipy.io as sio
import itk
from time import time

from VascGraph.Skeletonize import GenerateGraph 
from VascGraph.Skeletonize import ContractGraph
from VascGraph.Skeletonize import RefineGraph


class ScanLines:
    
    def __init__(self, raw, scales=1, tolerance=.1):
       
        # read array
        self.Im=itk.GetImageFromArray(raw.astype('float32'))
        self.Im.UpdateOutputData()
        
        self.tolerance=tolerance
        self.scales=scales
        
        self.VessMap=[]
        self.BinaryMap=None
        self.Skeleton=None
        self.Graph=None
        self.Lines=None

        #####
        # private
        #####

    def GetVessImage(self, im, sigma=1):
        
        # apply hessian
        dimension = im.GetImageDimension()
        InputImageType = type(im)
        OutputPixelType = itk.UC
        OutputImageType = itk.Image[OutputPixelType, dimension]
        cast = itk.CastImageFilter[InputImageType, OutputImageType].New(im)
        hessianFilter = itk.HessianRecursiveGaussianImageFilter.New(cast.GetOutput())
        hessianFilter.SetSigma(sigma)
        hessianFilter.Update()
        
        # apply vesselness filter
        hessOutput = hessianFilter.GetOutput()
        dimension = hessOutput.GetImageDimension()
        InputImageType = type(hessOutput)
        OutputPixelType = itk.F
        OutputImageType = itk.Image[OutputPixelType, dimension]
        vess=itk.HessianToObjectnessMeasureImageFilter[InputImageType, 
                                                         OutputImageType].New(hessOutput)
        vess.Update()
        vessOutput = vess.GetOutput()    
        self.VessMap.append(vessOutput)  
        return vess
        
    def GetBinaryImage(self, im, sigma=1):   
        
        vess=self.GetVessImage(im, sigma=sigma)     
        vessOutput = vess.GetOutput()    
        dimension = vessOutput.GetImageDimension()
        InputImageType = type(vessOutput)
        OutputPixelType = itk.F
        OutputImageType = itk.Image[OutputPixelType, dimension]
        vessOutput = itk.CastImageFilter[InputImageType, OutputImageType].New(vessOutput)
        
        # norm and thresholding from vessOutput
        norm=itk.NormalizeImageFilter.New(vessOutput)
        norm.Update()
        binary=itk.BinaryThresholdImageFilter.New(norm)
        binary.SetLowerThreshold(0)
        binary.Update()
                  
        return binary


    def RemoveSmallObjects(self, binary, thr_size=5):
        
        # norm and thresholding from vessOutput
        binary=itk.BinaryThresholdImageFilter.New(binary.GetOutput())
        binary.SetLowerThreshold(1)
        binary.Update()    
        
        # get labels and remove small ones
        binaryOutput=binary.GetOutput()
        labels=itk.BinaryImageToLabelMapFilter.New(binaryOutput)
        labels.Update()
        labelsOutput=labels.GetOutput()
        nLabels=labelsOutput.GetNumberOfLabelObjects()
        
        size=[]
        for i in range(nLabels):   
            size.append(int(labelsOutput.GetNthLabelObject(i).Size()))
        
        labelsToRemove=np.where(np.array(size)<thr_size)[0]
        for i in labelsToRemove:
            try:
                labelsOutput.RemoveLabel(i+1)
            except:
                pass
        seg=itk.LabelMapToBinaryImageFilter.New(labelsOutput)
        seg.Update()
        
        return seg


    def GetBinaryMap(self):
        
        sigmas=[1,3,5,7,9,13]
        fillRadius=1
        
        # get binary images at different ranks        
        binaryImages=[]       
        for i in range(self.scales):           
            binaryImages.append(self.GetBinaryImage(self.Im,    
                                        sigma=sigmas[i]))  
        
        # add binary maps 
        if self.scales>1:
            
            for i in range(len(binaryImages)):           
                if i>0:
                    if i==1:
                        add=itk.AddImageFilter.New(binaryImages[i-1], binaryImages[i])
                        add.Update()                
                    else:                            
                        add=itk.AddImageFilter.New(add.GetOutput(), binaryImages[i])
                        add.Update()
        else:
            
            add=binaryImages[0]
                            
       
        # remove small objects  
        refined=self.RemoveSmallObjects(add, thr_size=5)    

        # fill holes
        refined=itk.VotingBinaryIterativeHoleFillingImageFilter.New(refined.GetOutput());
        refined.SetRadius(fillRadius)
        refined.Update()
        
        self.BinaryMap=refined.GetOutput()


    def GetSkeleton(self):
        
        if self.BinaryMap:
            
            # get skeleton
            skl=itk.BinaryThinningImageFilter.New(self.BinaryMap)
            skl.Update()    
            
            self.Skeleton=skl.GetOutput()
    
    def GetGraphFromSkl(self):

        if self.Skeleton is not None:
            
            skl=itk.GetArrayFromImage(self.Skeleton) 
            skl=np.swapaxes(skl,0,1).astype('bool')
            
            # generate graph
            pos=np.array(np.where(skl>0)).T
            
            graph=nx.Graph()
            graph.add_nodes_from(range(len(pos)))
            for ind,p in enumerate(pos):
                graph.node[ind]['pos']=[p[0],p[1],0]
        
        
            t=sp.spatial.cKDTree(pos.astype('float'))
            nbrs=[t.query(i, k=3, distance_upper_bound=1.5)[1] for i in pos]
            nbrs=np.array(nbrs)
            
            c1=nbrs[:,(0,1)]
            c2=nbrs[:,(0,2)]    
            c=np.vstack((c1,c2))
            
            graph.add_node(graph.number_of_nodes())
            graph.add_edges_from(c)
            graph.remove_node(graph.number_of_nodes()-1)
            
            self.Graph=graph

    def GetGraphFromSeg(self):
        
        sampling=1.0
    
        speed_param=0.001 # speed of contraction process (smaller value-->faster dynamics)
        dist_param=0.01 # [0,1] controls contraction based on connectivitiy in graph
        med_param=1.0 # [0,1] controls contraction based on distance map (obtained from binary image)
        #-------------------------------------------------------------------------#
        # hyper parameters (modification on those is not suggested!)
        #-------------------------------------------------------------------------#
        #contraction
        degree_threshold=10.0 # used to check if a node is a skeletal node
        clustering_resolution=1.0 # controls the amount of graph dicimation (due to clustering) at each step
        stop_param=0.005 # controls the convergence criterion
        n_free_iteration=10 #number of iteration without checking for convergence
        #refinement
        area_param=50.0 # area of polygens to be decimated 
        poly_param=10 # number of nodes forming a polygon        
        
        if self.BinaryMap is not None:
            
            binmap=itk.GetArrayFromImage(self.BinaryMap) 
            binamap=np.swapaxes(binmap,0,1).astype('bool')
            label=np.array(binmap[None,:])
            label=np.rollaxis(label,0,3)
            
            generate=GenerateGraph(label)
            generate.UpdateGridGraph(Sampling=sampling)
            graph=generate.GetOutput()
            
            contract=ContractGraph(graph)
            contract.Update(DistParam=dist_param, 
                            MedParam=med_param, 
                            SpeedParam=speed_param, 
                            DegreeThreshold=degree_threshold, 
                            StopParam=stop_param,
                            NFreeIteration=n_free_iteration)
            gc=contract.GetOutput()
            
            refine=RefineGraph(gc)
            refine.Update(AreaParam=area_param, 
                          PolyParam=poly_param)
            gr=refine.GetOutput()  
      
            self.Graph=gr
         
    def GetLines(self):
        
        diam=self.diam 
        length=self.length         
        if self.Graph:
            
            graph=self.Graph.copy()
            
            
            # get potential lines
            tolerance=self.tolerance
            smooth_itr=1
            seg_len=4
      
            nb=[graph.neighbors(i) for i in graph.nodes()]
            nodesToRemove1=[k for k,i in enumerate(nb) if len(i)>2]
            nodesToProcess=[k for k,i in enumerate(nb) if len(i)==2]
            
            # average position
            for i in range(smooth_itr):
                ownpos=np.array([graph.node[i]['pos'] for i in nodesToProcess])
                otherpos=np.array([[ graph.node[k]['pos']  for k in graph.neighbors(i)] for i in nodesToProcess])
                posNew=(ownpos.astype('float')+otherpos[:,0,:]+otherpos[:,1,:])/3   
                for ind, node in enumerate(nodesToProcess):
                    graph.node[node]['pos']=posNew[ind]
            
            otherpos=np.array([[ graph.node[k]['pos']  for k in graph.neighbors(i)] for i in nodesToProcess])
            indNodesToRemove2=isSklNodes(posNew, otherpos, thr=tolerance)==False  
            nodesToProcess=np.array(nodesToProcess)
            nodesToRemove2=nodesToProcess[indNodesToRemove2]
            
            graph.remove_nodes_from(nodesToRemove1)
            graph.remove_nodes_from(nodesToRemove2)
            graph=fixG(graph)
                
            # get graph segments
            components=list(nx.connected_components(graph))
            components=[i for i in components if len(i)> seg_len]       
            segments=[[i for i in j if len(graph.neighbors(i))==1] for j in components]        
            segments=np.array(segments)
            
            # remove segments with big diam
            segmentsToRecall=[]
            for idx, i in enumerate(segments):
                if graph.node[i[0]]['d']<diam and graph.node[i[1]]['d']<diam:
                    segmentsToRecall.append(idx)            
            segments=segments[segmentsToRecall,:]
       
            lines=[[np.array(graph.node[j]['pos']) for j in i] for i in segments]  
            lines=np.array(lines)
            
            # retrieve with length constrains
            
            norm=np.linalg.norm(lines[:,0,:]-lines[:,1,:], axis=1)
            mid=np.mean(lines, axis=1)
            unitV=(lines[:,0,:]-lines[:,1,:])/norm[:,None]
            
            IndLinesToReduce=np.where(norm>length)
            LinesReduced=np.array([mid[IndLinesToReduce]-length*unitV[IndLinesToReduce]/2.0,
                                    mid[IndLinesToReduce]+length*unitV[IndLinesToReduce]/2.0])
            LinesReduced=np.swapaxes(LinesReduced,0,1)
            lines[IndLinesToReduce]=LinesReduced
       
            self.Lines=lines
            
        
    #######
    # Public
    #######
    
    def CalcBinaryMap(self, scales=1):
        self.scales=scales
        self.GetBinaryMap() 
        
    def CalcSkeleton(self):
        self.GetSkeleton()
        
    def CalcGraphFromSeg(self):
        self.GetGraphFromSeg()

    def CalcGraphFromSkl(self):
        self.GetGraphFromSkl()
        
    def CalcLines(self, diam=5, length=15, tolerance=.1):
        self.diam=diam
        self.length=length
        self.tolerance=tolerance
        self.GetLines()

        
    def GetOutputBinaryMap(self):
        
        if self.BinaryMap is not None:
            
            im=itk.GetArrayFromImage(self.BinaryMap) 
            im=np.swapaxes(im,0,1).astype('bool')
            
            return im       

    def GetOutputVessMap(self):
        
        if len(self.VessMap)>0:
            
            vessmap=[]
            
            for i in self.VessMap:
                
                vess=itk.GetArrayFromImage(i) 
                vessmap.append(np.swapaxes(vess,0,1).astype('float'))   
                
            return vessmap  
                    
    def GetOutputSkeleton(self):
        
        if self.Skeleton is not None:
            
            skl=itk.GetArrayFromImage(self.Skeleton) 
            skl=np.swapaxes(skl,0,1).astype('bool')
            
            return skl

    def GetOutputGraph(self):
        
        if self.Graph is not None: return self.Graph
 
    def GetOutputLines(self):
        
        if self.Lines is not None: return self.Lines


if __name__=='__main__':    
  
    pass

