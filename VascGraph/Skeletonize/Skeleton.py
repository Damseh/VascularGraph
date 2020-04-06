#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:26:17 2020

@author: rdamseh
"""

from VascGraph.Skeletonize import GenerateGraph, ContractGraph, RefineGraph
from VascGraph.GeomGraph import Graph as EmptyGraph
from VascGraph.Tools.CalcTools import fixG

import numpy as np
import scipy.ndimage as image
import networkx as nx

def Decompose(s, size=100):
    
    try:
        import skimage
    except:
        print('  \'scikit-image\' must be installed to run this function!')
        return 
    
    shape = s.shape
    s1, s2, s3 = shape
    
    pad1, pad2, pad3 = [s1%size, s2%size, s3%size]
    s=skimage.util.pad(s, ((0,size-pad1), (0,size-pad2), (0,size-pad3)), constant_values=((0,0),(0,0),(0,0)))
    patches=skimage.util.view_as_windows(s, (size, size, size), step=size)

    ind1, ind2, ind3 =np.meshgrid(list(range(patches.shape[0])),
                           list(range(patches.shape[1])),
                           list(range(patches.shape[2])), indexing='ij')
    patchesid = [(i1, i2, i3) for i1, i2, i3 in zip(ind1.ravel(), ind2.ravel(), ind3.ravel())]
    
    return patches, patchesid

def ContractExt(graph, niter, DistParam=0, 
                              MedParam=1, 
                              SpeedParam=.05, 
                              DegreeThreshold=10, 
                              ClusteringResolution=1):
  
    for k in graph.GetNodes():
        if graph.node[k]['ext']==0:
            graph.node[k]['ext']=1
        else:
            graph.node[k]['ext']=0 
                
    contract=ContractGraph(graph, freeze_ext=True)
    for i in range(niter):
        try:
            contract.UpdateOneStep(DistParam=DistParam, 
                                   MedParam=MedParam, 
                                   SpeedParam=SpeedParam, 
                                   DegreeThreshold=DegreeThreshold, 
                                   ClusteringResolution=ClusteringResolution,
                                   update_positions=True)
            
            contract.UpdateTopologyOneStep()
        except: pass    

def activate_parallel():
    
    try:
        import ray
    except:
        print('  \'ray\' must be installed to run patch-based contraction in parallel!')
        
    # test graphing
    @ray.remote
    def GraphParallel(patch, niter, Sampling=1,
                                    DistParam=0, 
                                    MedParam=1, 
                                    SpeedParam=.05, 
                                    DegreeThreshold=10, 
                                    ClusteringResolution=1):
        
        generate=GenerateGraph(Label=patch>0, DisMap=patch, label_ext=True)
        generate.UpdateGridGraph(Sampling=Sampling)
        graph=generate.GetOutput()
        area=generate.Area 
        try:
            contract=ContractGraph(graph, freeze_ext=True)
            for i in range(niter):
                contract.UpdateOneStep(DistParam=DistParam, 
                                       MedParam=MedParam, 
                                       SpeedParam=SpeedParam, 
                                       DegreeThreshold=DegreeThreshold, 
                                       ClusteringResolution=ClusteringResolution,
                                       update_positions=True)
                
                contract.UpdateTopologyOneStep()  
            gc=contract.GetOutput() 
            gc.Area=area
        except:
            gc=None
            print('--Cannot contract the graph of a patch. \n    Number of nodes: '+str(graph.number_of_nodes()))
        return gc
    
    return GraphParallel

def GraphSerial(patch, niter, Sampling=1,
                                DistParam=0, 
                                MedParam=1, 
                                SpeedParam=.05, 
                                DegreeThreshold=10, 
                                ClusteringResolution=1):
    
    generate=GenerateGraph(Label=patch>0, DisMap=patch, label_ext=True)
    generate.UpdateGridGraph(Sampling=Sampling)
    graph=generate.GetOutput()
    area=generate.Area 
    try:
        contract=ContractGraph(graph, freeze_ext=True)
        for i in range(niter):
            contract.UpdateOneStep(DistParam=DistParam, 
                                   MedParam=MedParam, 
                                   SpeedParam=SpeedParam, 
                                   DegreeThreshold=DegreeThreshold, 
                                   ClusteringResolution=ClusteringResolution,
                                   update_positions=True)
            
            contract.UpdateTopologyOneStep()  
        gc=contract.GetOutput() 
        gc.Area=area
    except:
        gc=None
        print('--Cannot contract the graph. \n    Number of nodes: '+str(graph.number_of_nodes()))
    return gc

def AddPos(g, add):
    if g is not None:
        for node in g.GetNodes():
            g.node[node]['pos']=g.node[node]['pos']+add

class Skeleton:
    
    '''
    class used to provide a final skelton given input binary label
        it is based on objects from GenerateGraph, ContractGraph and RefineGraph classes
    '''
    
    def __init__(self, label, 
                 speed_param=0.1, 
                 dist_param=0.5, 
                 med_param=0.5, 
                 sampling=1, 
                 degree_threshold=10.0,
                 clustering_resolution=None,
                 stop_param=0.005,
                 n_free_iteration=5,
                 area_param=50.0,
                 poly_param=10):

        # generate
        self.label=label
        self.sampling=sampling        


        #contraction
        self.speed_param=speed_param
        self.dist_param=dist_param
        self.med_param=med_param
        self.degree_threshold=degree_threshold # used to check if a node is a skeletal node
        if clustering_resolution is not None:
            self.clustering_resolution=clustering_resolution # controls the amount of graph dicimation (due to clustering) at each step
        else:
            self.clustering_resolution=sampling
        self.stop_param=stop_param # controls the convergence criterion
        self.n_free_iteration=n_free_iteration #number of iteration without checking for convergence
        
        #refinement
        self.area_param=area_param # area of polygens to be decimated 
        self.poly_param=poly_param # number of nodes forming a polygon    

    def Update(self, label=None, ret=False): 

        '''
        generate a graph based on the input label image
        
        method: generate initial geometry --> contract graph --> refine graph
        
        
        @article{damseh2019laplacian,
            title={Laplacian Flow Dynamics on Geometric Graphs for Anatomical Modeling of Cerebrovascular Networks}, 
            author={Damseh, Rafat and Delafontaine-Martel, Patrick and Pouliot, Philippe and Cheriet, Farida and Lesage, Frederic}, 
            journal={arXiv preprint arXiv:1912.10003}, year={2019}}
        
        @article{damseh2018automatic,
            title={Automatic Graph-Based Modeling of Brain Microvessels Captured With Two-Photon Microscopy}, 
            author={Damseh, Rafat and Pouliot, Philippe and Gagnon, Louis and Sakadzic, Sava and Boas, 
                    David and Cheriet, Farida and Lesage, Frederic}, 
            journal={IEEE journal of biomedical and health informatics}, 
            volume={23}, 
            number={6}, 
            pages={2551--2562}, 
            year={2018}, 
            publisher={IEEE}} 
                    
        '''

        # ---------- generate -----------#
        if label is None: 
            generate=GenerateGraph(self.label)
        else:
            generate=GenerateGraph(label)
            
        generate.UpdateGridGraph(Sampling=self.sampling)
        graph=generate.GetOutput()    
    
        # ---------- contract -----------#
        contract=ContractGraph(graph)
        contract.Update(DistParam=self.dist_param, 
                        MedParam=self.med_param, 
                        SpeedParam=self.speed_param, 
                        DegreeThreshold=self.degree_threshold, 
                        StopParam=self.stop_param,
                        NFreeIteration=self.n_free_iteration)
        gc=contract.GetOutput()    
    
        # ---------- refine -----------#
        refine=RefineGraph(gc)
        refine.Update(AreaParam=self.area_param, 
                      PolyParam=self.poly_param)
        gr=refine.GetOutput() 
        gr=fixG(gr)        
        
        # ----- return ----#
        if ret:
            return gr

        else:
            self.Graph=gr

        
    def UpdateWithStitching(self, size, 
                            niter1=10, niter2=5, 
                            is_parallel=False, n_parallel=5,
                            ret=False):
        
        '''
        this funtion allow to generate graphs as follows:
            1) image patching -->  2) patch-based contraction (fixing boundary nodes) 
            --> 3) graph stitching --> 4) boundary contraction --> 5) global contraction --> refinement 
        
        it is helpful when graphing large inputs.
        
        Inputs:
            size: dimention of a 3D patch --> [size, size, size] 
            niter1: number of contraction iterations on patches 
            niter2: number of contraction iterations on boundary nodes 
            is_parallel: if True, patch-based contraction will run in parallel using 'ray'
            n_parallel: number of parallel processes (note: for limited RAM memory, 'n_parallel' should be smaller) 
            ret: if True, this function will return the output graph
        '''
        try:
            from sklearn import neighbors
        except:
            print('  \'scikit-learn\' must be instaled to run this funtion!')        
        
        if is_parallel:
            
            try:
                import ray
            except:
                print('  \'ray\' must be installed to run patch-based contraction in parallel!')            
            
            GraphParallel=activate_parallel()  
            
            
            
            
        # obtain distance map
        self.label=image.morphology.distance_transform_edt(self.label)
        
        # patching
        print('--Extract patches ...')
        patches, patchesid=Decompose(self.label, size=size) # extract patches
        patches_shape=[patches.shape[0], 
                       patches.shape[1],
                       patches.shape[2]]
            
        print('--Obtain semi-contracted graphs from patches ...')
        # run contraction avoiding boundary nodes for each patch
        graphs=[]
        inds=np.arange(0,len(patchesid),n_parallel)
        patchesid_=[patchesid[ind:ind+n_parallel] for ind in inds]
        
        
        for inds in patchesid_:
            
            if is_parallel: # in parallel
                ray.init()
                subpatches=[ray.put(patches[ind]) for ind in inds]
                subgraphs=[GraphParallel.remote(patch, niter=niter1, 
                                                Sampling=self.sampling,
                                                DistParam=self.dist_param, 
                                                MedParam=self.med_param, 
                                                SpeedParam=self.speed_param, 
                                                DegreeThreshold=self.degree_threshold, 
                                                ClusteringResolution=self.clustering_resolution) for patch in subpatches]
                subgraphs=[ray.get(g) for g in subgraphs]
                ray.shutdown()
                graphs.append(subgraphs)
           
            else: # in serial
                subpatches=[patches[ind] for ind in inds]
                subgraphs=[GraphSerial(patch, niter=niter1, 
                                                Sampling=self.sampling,
                                                DistParam=self.dist_param, 
                                                MedParam=self.med_param, 
                                                SpeedParam=self.speed_param, 
                                                DegreeThreshold=self.degree_threshold, 
                                                ClusteringResolution=self.clustering_resolution) for patch in subpatches]
                subgraphs=[g for g in subgraphs]
                graphs.append(subgraphs)
        graphs = [k1 for k in graphs for k1 in k] # uravel
        del patches 
        
        # adjust the position of graph nodes coming from each patch 
        area=np.sum([k.Area for k in graphs if k is not None])
        pluspos=(size)*np.array(patchesid)
        for plus, g in zip(pluspos, graphs):
            if g is not None:
                AddPos(g, plus) 
        
        
        print('--Combine semi-contracted graphs ...')
        fullgraph=EmptyGraph()
        nnodes=0
        for idx, g in enumerate(graphs):
            if g is not None:
                print('    graph id '+str(idx)+' added')
                nnodes+=fullgraph.number_of_nodes()
                new_nodes=nnodes+np.array(range(g.number_of_nodes()))
                mapping = dict(zip(g.GetNodes(), new_nodes))
                g=nx.relabel_nodes(g, mapping) 
                fullgraph.add_nodes_from(g.GetNodes())
                fullgraph.add_edges_from(g.GetEdges())
                for k in new_nodes: 
                    fullgraph.node[k]['pos']=g.node[k]['pos']
                    fullgraph.node[k]['r']=g.node[k]['r']
                    fullgraph.node[k]['ext']=g.node[k]['ext']
            else:
                print('    graph id '+str(idx)+' is None')        
            
        fullgraph= fixG(fullgraph)
        fullgraph.Area=area    
        del graphs
            
        print('--Stitch semi-contracted graphs ...')
        nodes=np.array([k for k in fullgraph.GetNodes() if fullgraph.node[k]['ext']==1])
        nodesid=dict(zip(range(len(nodes)), nodes))
        pos=np.array([fullgraph.node[k]['pos'] for k in nodes])
        pos_tree=neighbors.KDTree(pos)
        a = pos_tree.query_radius(pos, r=1.0)
        new_edges=[[(nodesid[k[0]], nodesid[k1]) for k1 in k[1:]] for k in a]
        new_edges=[k1 for k in new_edges for k1 in k]
        fullgraph.add_edges_from(new_edges)    
        
        del a
        del nodes
        del pos
        del new_edges
        del pos_tree

        print('--Contract ext nodes ...')
        ContractExt(fullgraph, niter=niter2, 
                              DistParam=self.dist_param, 
                              MedParam=self.med_param, 
                              SpeedParam=self.speed_param, 
                              DegreeThreshold=self.degree_threshold, 
                              ClusteringResolution=self.clustering_resolution)

        print('--Generate final skeleton ...')
        contract_final=ContractGraph(Graph=fullgraph)
        contract_final.Update(DistParam=self.dist_param, 
                            MedParam=self.med_param, 
                            SpeedParam=self.speed_param, 
                            DegreeThreshold=self.degree_threshold, 
                            StopParam=self.stop_param,
                            NFreeIteration=self.n_free_iteration)
        gc = contract_final.GetOutput()
        
        print('--Refine final skeleton ...')
        refine = RefineGraph(Graph=gc)
        refine.Update()
        gr = refine.GetOutput()
        gr=fixG(gr)
        
        # ----- return ----#
        if ret:
            return gr
        else:
            self.Graph=gr

    def GetOutput(self):
        try:
            return self.Graph
        except:
            print('Update first!')
        
    
