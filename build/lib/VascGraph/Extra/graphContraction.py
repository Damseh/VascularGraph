#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 15:08:02 2018

@author: rdamseh
"""
from util import *
from scipy import io as sio
from time import time
import timeit
import scipy.sparse as s
import scipy.sparse.linalg  as la

class graphContraction:
       
    
    def __init__(self, label=None, G=None, animate=None, camGraph=None):
        
        if label is None and G is None:
            raise IOError('Missing input: Binary mask or initial graph.')
             
        if G is not None:
            self.G_init=G 
            self.check(self.G_init)
            
        self.label=label
        self.shape=np.shape(self.label)
        self.__computeMediality()
        
        # animation param
        self.animate=animate
        self.camGraph=camGraph
                # if animation if needed: create figure
        
        if self.animate:
           
            mlab.figure(bgcolor=(1,1,1), size=(1024,1024)) 
            visStack(self.label, opacity=.15, color=(.7,.7,.7), mode='same')
            e=mlab.get_engine()
            self.s=e.scenes[0]
        
        #parameters
        self.thr=5
        self.ln=10
    
    # private functions
    
    def __computeMediality(self):
        
        '''
        Compute mediality from the binary mask
        '''
        
        #get mediality values 
        t0=time()
        dist_map=getDist(self.label)  #get distance map      
        dist_map=filt.maximum_filter(dist_map, size=(3,3,3)) # apply max filter
        self.dist_map=dist_map
        print('Time to compute distance map: '+str(time()-t0))

    
    def __assignMediality(self, G=None, nodes=None):
        
        '''
        Assign medialiy values to graph nodes based
        on the Medilaity map       
        '''
        if G:
            pass
        else:
            G=self.G_contracted
                
        if nodes is not None:
            pass
        else:
            nodes=G.nodes()
            
        for i in nodes:
            
            pos=tuple(G.node[i]['pos'].astype(int).tolist())
            G.node[i]['node']=False # not in skeleton  

            if pos[0]<self.shape[0] and pos[1]<self.shape[1] and pos[2]<self.shape[2]:
                
                d=self.dist_map[pos]               
                
                if d<1: 
                    d=0
                else:
                    G.node[i]['dd']=d   
                    G.node[i]['d']=d
                
            else:
                G.node[i]['dd']=0   
                G.node[i]['d']=0
        
    
    def __updateTopo(self, n, eps=2, refine=False):
        
        '''
        update topology of the graph after 
        a contraction step
        '''
        
        if refine==False:
            G=self.G_contracted
        else:
            G=self.G_refined
        
        print('Update topology...') 
        #obtain for the nodes to ungergo surgury 
        pos=[G.node[i]['pos'] for i in n]           
        pos=np.array(pos).astype('float')        

        #get skl nodes
        r=set(G.nodes()).difference(set(n))
        r=list(r)
        
        print('Perform clustering...')
        # cluster vertices of graph
        # get centroids and verticies labels blonging to that centroids   
        #centroids, poin, poin_ind_= DBSCAN_cluster(pos, eps=eps)
        
        centroids, poin, poin_ind_= DBSCAN_cluster(pos, eps=eps)
        
        #map indicies of poin_ind to indices in nodes_to_process 
        #(to get the origional indices of nodes in the graph)
        poin_ind=[[n[j] for j in i]  for i in poin_ind_]
          
        #check num of clusters and exit iteration if it returns false
        n_diff=0
        unique_centroids=[]
        unique_poin_ind=[]
        single_clusters=[]
        
        for i in range(len(poin_ind)):
            if len(poin_ind[i])>1:
                n_diff=n_diff+1
                unique_centroids.append(centroids[i])
                unique_poin_ind.append(poin_ind[i])
            else:
                single_clusters.append(poin_ind[i][0])
                
        print('Number of new clusters: '+str(n_diff))
       
        #nodes to be kept in the graph
        nodes_to_keep=list(set(r).union(set(single_clusters)))
        
        #calculate med_values for the new centroids
             
        med_vals_idx= [ tuple(map(int,i)) for i in  centroids] 
        # to avoid index from outside
        cen_med_vals= [ self.dist_map[i] if 
                       i[0] < self.shape[0] and
                       i[1] < self.shape[1] and
                       i[2] < self.shape[2] else 1 for i in med_vals_idx]
                
       
        print('Perform toplogical surgery...')
        # update graph with new connections based on the new vertices               
        G=self.__connectSrg(G, 
                     pnts=unique_poin_ind,  
                     centroids=unique_centroids,
                     d_map=cen_med_vals,
                     skl_nodes=nodes_to_keep)
 
        if refine==False:
            self.G_contracted=G
        else:
            self.G_refined=G
            
            
        return n_diff 
 
    def __updateTopo2(self, n=None, refine=False):
        
        '''
        update topology of the graph after 
        a contraction step
        '''
        
        if refine==False:
            G=self.G_contracted
        else:
            G=self.G_refined
        
        print('Update topology...') 
               
        #obtain for the nodes to ungergo surgury 
        
        if n:
            pos=np.array([G.node[i]['pos'] for i in n])
        else:
            pos=self.pos_to_process 
        
        print('Perform clustering...')
        
        # cluster vertices of graph   
        if pos.shape[0]>0:
            
            centroids, poin, poin_ind_= assignPixelClusters(pos) 
            
            #map indicies of poin_ind to indices in nodes_to_process 
            #(to get the origional indices of nodes in the graph)
            poin_ind=[[n[j] for j in i]  for i in poin_ind_]
            
            clusterd_nodes=[j for i in poin_ind for j in i]
    
            #check num of clusters and exit iteration if it returns false
            n_diff=len(centroids)   
            print('Number of new clusters: '+str(n_diff))
           
            
            #nodes to be kept in the graph
            nodes_to_keep=list(set(self.nodes).difference(set(clusterd_nodes)))
            
            if n_diff>0:
                
                print('Perform toplogical surgery...')
                # update graph with new connections based on the new vertices               
                G=self.__connectSrg2(G, 
                             pnts=poin_ind,  
                             centroids=centroids,
                             skl_nodes=nodes_to_keep)
         
                if refine==False:
                    self.G_contracted=G
                else:
                    self.G_refined=G

                return n_diff 
        else:
            print('No nodes to process!')
            return 0



    def __connectSrg2(self, G, pnts, centroids, skl_nodes):       
            
        '''
        Perform topological surgery based on the new clusters
        
        Input:
            
        '''
        
        # add new indices to the graph and assign postions and med values for
        #cluster containing more than one node
        new_nodes=list(1+np.array(range(len(centroids)))+np.max(G.nodes()))
        G.add_nodes_from(new_nodes)
        
        new_d=[np.max([ G.node[j]['d'] for j in i ]) for i in pnts]
        
        for itr, i in enumerate(new_nodes):
            G.node[i]['pos']=centroids[itr]
            G.node[i]['node']=False
            G.node[i]['d']=new_d[itr]
        
        #get neighbours of neighbourse for points in clusters
        nbrs_of_nbrs=[]           
        for itr, i in enumerate(new_nodes):        
            
            # get neighbours of veticies belonging to this cluster
            nbrs_=list(pnts[itr])
            
            #obtain neighbors of neighbors
            nbrs_of_nbrs1=[G.neighbors(j) for j in nbrs_]                      
            
            # flatten and extract a set (no duplicates)
            nbrs_of_nbrs2=[j for k in nbrs_of_nbrs1 for j in k]
            nbrs_of_nbrs2=list(set(nbrs_of_nbrs2).difference(set(nbrs_)))
            nbrs_of_nbrs.append(nbrs_of_nbrs2)          
            
        #build new conenctions and assign to the graph    
        new_connections1=[[i,j] 
                        for itr,i in enumerate(new_nodes) 
                        for j in nbrs_of_nbrs[itr]]
        G.add_edges_from(new_connections1) # assign connections 1              
                    
        #remove origional vetices used for clustering
        nodes_to_keep=set(new_nodes).union(set(skl_nodes))
        other_nodes=list(set(G.nodes()).difference(nodes_to_keep))
        
        
        #map other_nodes with their corresponding cluster
        other_nodes_cluster=dict(zip(other_nodes,[0 for i in other_nodes]))
        for centid, i in zip(new_nodes, pnts):
            for j in i:
                other_nodes_cluster[j]=centid
        
          
        #find clusters with direct connection between their nodes  
        new_connections2=[]
        for centid, i in zip(new_nodes, nbrs_of_nbrs):
            for j in i:
                try:
                    new_connections2.append([centid, other_nodes_cluster[j]])                  
                except:
                    pass
        G.add_edges_from(new_connections2)     
              
        
        #remove unwanted nodes            
        G.remove_nodes_from(other_nodes)
        
        print('Number of graph nodes: '+str(G.number_of_nodes())+'\n')
        
        #if nx.number_connected_components(G)>1:
            #print('Disconnected components created!')
        
        return G    
    
    
    def __connectSrg(self, G, pnts, centroids, d_map, skl_nodes):       
            
        '''
        Perform topological surgery based on the new clusters
        
        Input:
            
        '''
        
        # add new indices to the graph and assign postions and med values for
        #cluster containing more than one node
        new_nodes=list(1+np.array(range(len(centroids)))+np.max(G.nodes()))
        G.add_nodes_from(new_nodes)
        
        new_d=[np.max([ G.node[j]['d'] for j in i ]) for i in pnts]
        
        for itr, i in enumerate(new_nodes):
            G.node[i]['pos']=centroids[itr]
            G.node[i]['dd']=d_map[itr]
            G.node[i]['node']=False
            G.node[i]['d']=new_d[itr]
        
        #get neighbours of neighbourse for points in clusters
        nbrs_of_nbrs=[]           
        for itr, i in enumerate(new_nodes):        
            
            # get neighbours of veticies belonging to this cluster
            nbrs_=list(pnts[itr])
            
            #obtain neighbors of neighbors
            nbrs_of_nbrs1=[G.neighbors(j) for j in nbrs_]                      
            
            # flatten and extract a set (no duplicates)
            nbrs_of_nbrs2=[j for k in nbrs_of_nbrs1 for j in k]
            nbrs_of_nbrs2=list(set(nbrs_of_nbrs2).difference(set(nbrs_)))
            nbrs_of_nbrs.append(nbrs_of_nbrs2)          
            
        #build new conenctions and assign to the graph    
        new_connections1=[[i,j] 
                        for itr,i in enumerate(new_nodes) 
                        for j in nbrs_of_nbrs[itr]]
        G.add_edges_from(new_connections1) # assign connections 1              
                    
        #remove origional vetices used for clustering
        nodes_to_keep=set(new_nodes).union(set(skl_nodes))
        other_nodes=list(set(G.nodes()).difference(nodes_to_keep))
        
        
        #map other_nodes with their corresponding cluster
        other_nodes_cluster=dict(zip(other_nodes,[0 for i in other_nodes]))
        for centid, i in zip(new_nodes, pnts):
            for j in i:
                other_nodes_cluster[j]=centid
        
          
        #find clusters with direct connection between their nodes  
        new_connections2=[]
        for centid, i in zip(new_nodes, nbrs_of_nbrs):
            for j in i:
                try:
                    new_connections2.append([centid, other_nodes_cluster[j]])                  
                except:
                    pass
        G.add_edges_from(new_connections2)     
              
        
        #remove unwanted nodes            
        G.remove_nodes_from(other_nodes)
        
        print('Number of graph nodes: '+str(G.number_of_nodes())+'\n')
        
        #if nx.number_connected_components(G)>1:
            #print('Disconnected components created!')
        
        return G    
    
    def __refineDiam(self, G=None, cutoff=5):
        
        if G:
            pass 
        else:
            G=self.G_refined.copy()
            
        
        cutoff=cutoff
        n=G.nodes()
        d=np.array([G.node[i]['d'] for i in n])
        
        # obtain neighbours of cutoff degree
        nbrs=nx.all_pairs_shortest_path(G, cutoff=cutoff)
        nbrs=[[nbrs[i][j] for j in nbrs[i].keys() if len(nbrs[i][j])>cutoff] for i in nbrs.keys()]
        nbrs=[np.ravel(i).tolist() for i in nbrs]
        nbrs=[list(set(i)) for i in nbrs]
        #
        # median filter
        dd=[d[i] for i in nbrs]
        dd=[np.median(i) for i in dd]
        #
        for i, diam in zip(n, dd):
            G.node[i]['d']=diam
        
        return G
        
        
        
    def __capture(self, G,  s, animate, iteration, camGraph):
            '''
            capture and save snapshot from mlab scene
            '''
            visG(fixG(G.copy()), radius=.7, color=(.3,.3,.7), gylph_r=2, gylph_c=(.7,.7,.3))
            #,jnodes_r=4, jnodes_c=(.7,.3,.3))
            setCam(self.camGraph)          
            mlab.savefig(animate+iteration, size=(1024,1024))
            s.children[1:2]=[]
            s.children[1:2]=[]  
    
    # public functions    
    def generateGraph(self, 
                      init_nodes=250000, 
                      connect=8,
                      sample=1,
                      portion=None):
        '''
        Genrate initial graph for the 
        binary mask
        '''
        self.sample=sample
        self.connect=connect
        self.init_nodes=init_nodes
        
        start=timeit.default_timer()
        
        # create intial graph from segmented image
        self.G_init=createGraphFromSeg5(self.label, sample=self.sample)

        # time
        self.n_initial_nodes=self.G_init.number_of_nodes()
        stop=timeit.default_timer()
        self.__t1=stop-start # time to create init geometry
        print('Time to create init geometry: '+str(self.__t1))
    
    
    def __check(self, G=None):
 
        '''
        Check the graph before contraction to obtain 
        the potential nodes to be processed
        '''
        if G:
            
            #remove single nodes
            nodes_to_remove=[]
            for i in G.nodes():
                if len(G.neighbors(i))==0:
                    nodes_to_remove.append(i)
            G.remove_nodes_from(nodes_to_remove)

            n=np.array(G.nodes())
            self.nodes=n 

            p=np.array([G.node[i]['pos'] for i in n])
            self.pos=p


            nbrs=[G.neighbors(i) for i in n]
            self.nbrs=nbrs

            pn=[np.array([G.node[i]['pos'] for i in j]) for j in nbrs]
            self.nbrs_pos=pn 

            
            #chck1
            self.chck1=[len(i)>1 for i in nbrs]  # True in nbrs > 3
            
            #chck2 
            self.chck2=isSklNodes(p, pn, self.thr) # true if skeleton
            #mask=chck2
            
            mask=np.logical_and(self.chck1, self.chck2) # true if skeleton, false otherwise
            
            #nodes_to_process & skl nodes
            n_=n[mask==False]
            skl_nodes=n[mask]
            #n_=n
            #skl_nodes=[]
            
            self.nodes_to_process=n_
            
            self.pos_to_process=np.array([G.node[i]['pos'] for 
                                          i in self.nodes_to_process])
            
            self.skl_nodes=skl_nodes
            self.mask=mask
            self.degree=G.degree()
            self.mediality=np.array([G.node[i]['d'] for i in self.nodes])


    def __checkIter(self):
            '''
            check if to continue iteration or not based on the area of polygns
            '''            
            thr=self.thr_iter
            
            #find polys
            cyc=nx.cycle_basis(self.G_contracted) 
            
            area=0
            for l in range(self.ln):
                if l>2:
                    t=[k for k in cyc if len(k)==l]              
                    #positon of poly vertices
                    p=np.array([[self.G_contracted.node[j]['pos'] for j in i]  for i in t])
                    # get polys that pass the area condition        
                    area+=np.sum(cycleAreaAll(p))
            
            
            chck=area>self.thr_iter
            
            return chck, area

    def __applyContraction(self):
    
        speed_param=self.speed_param
        dis_param=self.dis_param
        med_param=self.med_param
        deg_param=self.deg_param
        n=self.nodes
        
        # assign the node id in the graph with its index in 'n' 
        n_=range(len(n))
        n_idx=dict(zip(n, n_)) 
        
        p=np.array(self.pos)
        nbrs=self.nbrs
        pn=self.nbrs_pos
        skl=self.skl_nodes
        mask=self.mask
        degree=self.degree
        lens=[self.degree[i] for i in n]
        
        mediality=self.mediality
        medd=mediality
        
        #norm_medd=mediality/np.max(mediality)
        #speed_vals=(1-norm_medd)*(speed_param[1]-speed_param[0])+speed_param[0]
        
        print('Solve linear system ...') 
        
        #compute wieghts
        t0=time()
              
        nodes_degree=[[degree[i] for i in j] for j in nbrs]
        nodes_degree=np.array(nodes_degree)
        nodes_degree, msk=numpy_fill(nodes_degree, lens)
        deg=nodes_degree.astype(float)/np.sum(nodes_degree, axis=1)[:,None]
        
        pn=np.array(pn)
        pn, msk=numpy_fill(pn, lens, 3)
        dis0=np.linalg.norm(p[:,None]-pn, axis=2)*msk#*nodes_degree
        dis1=np.sum(dis0, axis=1)
        dis=dis0/dis1[:,None]        
        
#        dis0=msk*1.0 # important: should be float
#        dis1=np.sum(dis0, axis=1)
#        dis=dis0/dis1[:,None]    

        nbrss=np.array(nbrs)
        nbrss, msk=numpy_fill(nbrss, lens) # padded numpy array to account for diff row sizes
        # switch between nodes index in graph and their index in order
        nodes_idx=[n_idx[i] for i in nbrss[msk].astype(int)]
        
        
        med0=np.zeros_like(nbrss)
        med0[msk]=medd[nodes_idx] # fill mediality values
        med1=np.sum(med0, axis=1) 
        med=med0/med1[:,None]

        
        print('Calc values: '+str(time()-t0))
        
    
        # construct laplacian operator
        # build sparse matrices a and b
        t0=time()

        ind1=np.zeros_like(nbrss)
        ind1=ind1+np.array(range(len(n)))[:,None]
        ind1=ind1[msk].astype(int).tolist()  
        ind2=nodes_idx
        
        self.ind1=ind1
        self.ind2=ind2
        
        t0=time()

        a=s.lil_matrix((len(n)*4,len(n)))
    
        dis_param=dis_param
        vals_dis=dis[msk]
      
        a[ind1,ind2]=vals_dis*dis_param
        a[n_,n_]=-1*dis_param         
 
       
        med_param=med_param
        vals_med=med[msk]      
        ind11=np.array(ind1)+len(n)
        ind22=np.array(n_)+len(n)
        a[ind11.tolist(),ind2]=vals_med*med_param
        a[ind22.tolist(),n_]=-1*med_param        
 
    
        deg_param=deg_param
        vals_deg=deg[msk]       
        ind111=np.array(ind1)+2*len(n)
        ind222=np.array(n_)+2*len(n)
        a[ind111.tolist(),ind2]=vals_deg*deg_param
        a[ind222.tolist(),n_]=-1*deg_param      
        
        d=(mask==False)*speed_param+(mask)*10*speed_param # skl nodes will move at lower speed
        #d=np.ones_like(self.nodes)*speed_param
        
        ind111=np.array(range(len(n)))+3*len(n)
        a[ind111.tolist(), range(len(n))]=d


        t0=time()
        
        a=a.tocoo()
        b=np.vstack([np.zeros_like(p), np.zeros_like(p), 
                     np.zeros_like(p), p*np.array([d,d,d]).T]) # b matrix
        
        print('Build A & B: '+str(time()-t0))
        
        t0=time()
        px_ = s.linalg.lsqr(a, b[:,0], atol=1e-05, btol=1e-05)[0] 
        py_ = s.linalg.lsqr(a, b[:,1], atol=1e-05, btol=1e-05)[0]
        pz_ = s.linalg.lsqr(a, b[:,2], atol=1e-05, btol=1e-05)[0]
        p_=np.array([px_, py_, pz_]).T
        
        print('Solve: '+str(time()-t0))
    
        #update the graph
        for idx, i in enumerate(n):
            self.G_contracted.node[i]['pos']=p_[idx]

        return True


    def contractGraph2(self,
                       dis_param,
                       deg_param,
                       med_param,
                       speed_param, 
                       eps=2, 
                       thr=10,
                       thr_iter=100,
                       n_iter=1,
                       stop_thr=.01):
             
        # parameters
        self.speed_param = speed_param
        self.deg_param = deg_param
        self.dis_param = dis_param
        self.med_param = med_param        
        self.thr = thr
        self.stop_thr=stop_thr
        self.eps = eps
        self.G_contracted = self.G_init.copy() 
        self.n_iter=n_iter
        self.thr_iter=thr_iter
             
        self.iteration = 1
        self.numclusters=[]      
        self.runtime=[]
        chck=True

            
        #assign new mediality values to vertices
        self.__assignMediality()
        
        while chck or self.iteration<=self.n_iter:
                    
            start = timeit.default_timer()
            print('Apply contraction...')     
            print('Iteration: '+str(self.iteration))
 
            # setup area threshold
            if self.iteration==1:
                chck, area=self.__checkIter()
                print('Area: '+str(area))
                self.thr_iter=area*self.stop_thr  
     
            # examin skeleton like nodes
            self.__check(self.G_contracted) 

            nodes_to_proc=self.nodes_to_process.tolist()
            print('Number of nodes to be processed: '+str(len(nodes_to_proc)))
        
            
            # apply contraction
            cont=self.__applyContraction() 
            
            if cont==False:
                print('Converged!')
                break
            
            # cluster and update topology
            n_diff=self.__updateTopo2(nodes_to_proc)
            
            self.numclusters.append(n_diff) 
            
            stop = timeit.default_timer()
            self.runtime.append(stop-start)
            
            if self.animate:
                self.__capture(s=self.s, G=self.G_contracted, animate=self.animate, 
                               iteration='itr'+str(self.iteration)+'.png', camGraph=self.camGraph)
            
            if self.iteration>=self.n_iter:
                
                print('Check convergance ...')
                chck, area=self.__checkIter()
                
                if not chck:
                    print('Converged! Area is less than '+str(self.thr_iter))
                else:
                    print('Area: '+str(area))
            
                    
                
            self.iteration=self.iteration+1
            
            
        self.G_contracted=fixG(self.G_contracted)  
          


    def refineGraph(self, 
                    ln=10,
                    alpha=100.0,
                    diam_refine=5):   
    
        '''
        Refine the graph after the cotraction process by removing small polygons.
        
        Input:
            ln: maximum number of vertices in a polygon
            alpha: area threshold based on which the polygon removal decision is taken
        '''
        
        try:
            self.G_refined=self.G_contracted.copy()
        except:
            raise IOError('Apply contraction on the initial graph first.')
        
        try:
            self.ln=ln
            self.alpha=alpha
        except:
            raise IOError('The \'ln\' and \'alpha\' parameters are needed.')
        
        for i in self.G_refined.nodes():
            if len(self.G_refined.neighbors(i))==1:
                self.G_refined.node[i]['skl']=True
            else:
                self.G_refined.node[i]['skl']=False
        
        ln=10 #number of edges in polygons
        
        self.runtime2=[]

        while 1:
            
            start=timeit.default_timer()
            
            #find polys
            cyc=nx.cycle_basis(self.G_refined) 
            t=[k for k in cyc if len(k)<self.ln]
            
            #positon of poly vertices
            p=[[self.G_refined.node[j]['pos'] for j in i]  for i in t]   
            
            # get polys that pass the area condition        
            ar=[cycleArea(i) for i in p]        
            t=[i for i,j in zip(t,ar) if j <alpha]
            p=[i for i,j in zip(p,ar) if j <alpha]
            
            #check if polys are found
            if len(t)==0:
                break
            
            # centers of polygons
            c=[np.mean(i, axis=0) for i in p]                   
            steps=[.5*(i-j) for i,j in zip(p,c)] 
                          
            #unravel t and p and steps        
            t=[j for i in t for j in i]
            p=[j for i in p for j in i]
            steps=[j for i in steps for j in i]
           
            #get movment for polygons vertices
            mov=dict()
            for itr, i in enumerate(t):
            
                try:
                    mov[i]=np.vstack((mov[i], steps[itr]))
                except:
                    mov[i]=steps[itr]
            
            # select random step if multiple steps exist for a vertix
            for i in mov.keys():
                
                try:
                    nm=np.shape(mov[i])[1] # check if there is more than one movments
                    ind=np.random.randint(0, nm-1)
                    mov[i]= mov[i][ind]
                
                except:
                    pass
                             
            # update nodes positions
            for i in self.G_refined.nodes():
                pos=self.G_refined.node[i]['pos']
                try:
                    self.G_refined.node[i]['pos']=pos-mov[i]
                except:
                    pass
            
            n=list(set(t)) # here, t is unravels
            
            self.__check(G=self.G_refined)
            nm_clusters=self.__updateTopo2(n=n, refine=True)   
            
            stop=timeit.default_timer()
            self.runtime2.append(stop-start)
            
            if self.animate:
                self.__capture(s=self.s, G=self.G_refined, animate=self.animate, 
                               iteration='itr'+str(self.iteration)+'.png', camGraph=self.camGraph)
            
            self.iteration=self.iteration+1


        self.G_refined=prunG(self.G_refined)
        self.G_refined=fixG(self.G_refined)
        self.iteration=self.iteration+1
        
        if self.animate:
            self.__capture(s=self.s, G=self.G_refined, animate=self.animate, 
                           iteration='itr'+str(self.iteration)+'.png', camGraph=self.camGraph)
            
        self.G_refined=getFullyConnected(self.G_refined)  
        
        self.G_final=self.__refineDiam(cutoff=diam_refine)
        


            
  
   