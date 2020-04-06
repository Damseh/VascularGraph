#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 12:36:37 2019

@author: rdamseh
"""

import numpy as np
import scipy.ndimage as image
import networkx as nx
from scipy import sparse

def AssignToClusters(pos, resolution=1.0):
    '''
    Assigne the current nodes in graph with closest pixel
    
    Input:
        pos: position of the current graph nodes
        
    Output:
        centroids: Geometric position for the centers of the clusters as [x,y,z].
        clusters_pos: qeometric positin of the nodes grouped in eache cluster.
        clusters_points: The indecies of points gropued in each cluster.   
        
        '''
    pos=np.array(pos)/resolution    
    clusters_init=np.round(pos).astype(int)
    
    c, clusters_index, clusters_inverse, clusters_count=np.unique(clusters_init, 
                             axis=0, return_inverse=True, return_index=True, 
                             return_counts=True)  
    
    clusters=np.where(clusters_count>=1)[0]
    clusters_points=[np.where(clusters_inverse==i)[0] for i in clusters]
    
    pos=pos*resolution
    clusters_pos=[pos[i] for i in clusters_points]
    centroids=[np.mean(i,axis=0) for i in clusters_pos]


    return centroids, clusters_pos, clusters_points




def CycleArea(corners):
    n = len(corners) # of corners
    cross= [0.0, 0.0, 0.0]
    for i in range(n):
        j = (i + 1) % n
        crss=np.cross(corners[i],corners[j])
        cross=cross+crss
        #nrm+=np.linalg.norm(crss)
    area = np.linalg.norm(cross) / 2.0
    return area

def CycleAreaAll(corners):
        
    if len(corners):       
        n = np.shape(corners)[1] # of corners
        cross= np.zeros((np.shape(corners)[0], np.shape(corners)[2]))
        for i in range(n):
            j = (i + 1) % n
            crss=np.cross(corners[:,i], corners[:,j])
            cross=cross+crss
            #nrm+=np.linalg.norm(crss)
        area = np.linalg.norm(cross, axis=1) / 2.0
    else:
        return 0
    return area

def CheckNode(a, b, thr=0):
    
    '''
    A fucntion to check a certian node is 
    to be processec in next iteration. The condition is based on angles 
    between the edges shared with the node.
    
    Input:
        -a: coordinate of a graph node
        -b: coordinateds of 'a' neighbours
        -thr: angle threshold 

    Output:
        True if a is to be processed and Flase other wise
    '''
    
    p1=b[0]
    p2=b[1:]

    ap1 = p1 - a
    ap2 = p2 - a
    
    x1,y1,z1=ap1
    x2,y2,z2=ap2[:,0], ap2[:,1], ap2[:,2] 
    
    dot=x1*x2+y1*y2+z1*z2
    norm1=(x1**2+y1**2+z1**2)**.5
    norm2=(x2**2+y2**2+z2**2)**.5 
    
    mask=norm2>0 # to avoid Inf
    cosine_angle =  np.ones_like(norm2)   
    if norm1>0:
        cosine_angle[mask]=dot[mask]/(norm1*norm2[mask])    
    
    # fix 1
    notvalid=(cosine_angle<-1)|(cosine_angle>1)
    cosine_angle[notvalid]=0 # exclude this case (90 degree)

   
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
     
    thr1, thr2 = thr, 180-thr
    chck=(angle>thr1)&(angle<thr2)

    
    return not np.any(chck) #true if skel


def IsSklNodes(p, pn, thr=0):
          
    '''
    output a boolian array with True values incidicating skeletal nodes
    '''
    
    # check nodes to process
    chck=[CheckNode(i, j, thr) for i, j in zip(p, pn)]  
    return np.array(chck) # true if  skeleton


def fixG(G, copy=True):
    try:
        Oldnodes=G.GetNodes()
        new=range(len(Oldnodes))
        mapping={Oldnodes[i]:new[i] for i in new}
        G=nx.relabel_nodes(G, mapping, copy=copy)
    except:
        Oldnodes=G.GetNodes()
        new=range(len(Oldnodes))
        mapping={Oldnodes[i]:new[i] for i in new}
        G=nx.relabel_nodes(G, mapping, copy=not copy)   
    return G


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
        out = np.zeros((mask.shape[0], mask.shape[1], s), dtype=data.dtype)
        out[mask, :] = np.concatenate(data)

    else:
        out = np.zeros((mask.shape[0],mask.shape[1]), dtype=data.dtype)
        out[mask] = np.concatenate(data)

    return out.astype(float), mask   


def numpy_fill_sparse(data, lens, s=False):
    '''
    Pad an array with different row sizes
    Input:
        data: object array
        lens: length of each row of data
        s=length of each element in the row
    '''
    lens=np.array(lens)

    # Mask of valid places in each row
    mask = sparse.lil_matrix(np.arange(lens.max()) < lens[:, None])

    # Setup output array and put elements from data into masked positions
    if s==3:    
        
        outx = sparse.lil_matrix((mask.shape[0], mask.shape[1]))
        outy = sparse.lil_matrix((mask.shape[0], mask.shape[1]))
        outz = sparse.lil_matrix((mask.shape[0], mask.shape[1]))
        
        datax=np.array([[k1[0] for k1 in k2] for k2 in data])
        datay=np.array([[k1[1] for k1 in k2] for k2 in data])
        dataz=np.array([[k1[2] for k1 in k2] for k2 in data])

        outx[mask] = np.concatenate(datax)
        outy[mask] = np.concatenate(datay)
        outz[mask] = np.concatenate(dataz)
        
        return outx, outy, outz, mask  
    
    else:
        data=np.array([[k1 for k1 in k2] for k2 in data])
        out = sparse.lil_matrix((mask.shape[0], mask.shape[1]))
        out[mask] = np.concatenate(data)
        return out, mask  

def get_difference(a, b):
     
    # -----------
    # if containes only one row
    try: 
        a.shape[1]
    except:a=a[:, None].T
    
    try: 
        b.shape[1]
    except:b=b[:, None].T
    # ------------
    
    a=set([tuple(i) for i in a])
    b=set([tuple(i) for i in b])
    return np.array(list(a.symmetric_difference(b)))
    


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
        nbrs.append(G.GetNeighbors(i))
        
    for n, nb in zip(j_nodes, nbrs):
        if len(nb)==1:
            if nb[0] in j_nodes:
                G.remove_node(n)
    G=fixG(G)
    
    return G

def FindJuntionNodes(G, Bifurcation=[], mode='not_all'):
    
        
    nodes=set()
    
    # fix bifurcation if needed
    if type(Bifurcation)!=list:
        pass
    else:
        Bifurcation=[].append(Bifurcation)
    
    if 1 not in Bifurcation:
        Bifurcation.append(1)
   
    # if all bifurcation    
    if mode=='not_all':
        for i in Bifurcation:
            u={node for node in G.GetNodes() if len(G.GetNeighbors(node))==i}
            nodes=nodes.union(u) 
 
    # if only specified  bifurcation      
    else:
        u={node for node in G.GetNodes() if 
           (len(G.GetNeighbors(node))==1 or len(G.GetNeighbors(node))>2)}
        nodes=nodes.union(u) 

    return list(nodes)


def FullyConnectedGraph(G):
    
    
    # connected components
    graphs=list(nx.connected_component_subgraphs(G))
    s=0
    ind=0
    for idx, i in enumerate(graphs):
        if len(i)>s:
            s=len(i); ind=idx
    G=graphs[ind]      
    return G


def ConnectedComponents(G, max_n=0):
    
    # connected components
    graphs=list(nx.connected_component_subgraphs(G))

    ind=[]
    for idx, i in enumerate(graphs):
        if len(i)>max_n:
            ind.append(idx)
    
    Gs=[graphs[i] for i in ind]
    
    G=nx.compose_all(Gs)
    return G



def reduceG(G, j_only=False):
    
    cont=1
    idNodes,_=findNodes(G, j_only=j_only)
    
    while cont!=0:
    #        NodesToRemove=[]
    #        NodesToConnect=[]
        cont=0
        for i in G.GetNodes():
            k=G.GetNeighbors(i)
            if len(k)==2:
                if i not in idNodes:
                    G.remove_node(i)
                    G.add_edge(k[0],k[1])
                    cont=1
    return G


def findNodes(G, j_only=False):
    
    nodes=[]
    ind=[]
    
    if j_only:      
        
        for i in G.GetNodes():
            if len(G.GetNeighbors(i))==3:
                nodes.append(G.node[i]['pos'])
                ind.append(i)      
    else:
               
        for i in G.GetNodes():
            if len(G.GetNeighbors(i))!=2:
                nodes.append(G.node[i]['pos'])
                ind.append(i)
                
    try:       
        nodes=[i.tolist() for i in nodes]
    except:
        pass

    return ind, nodes


def adjustGraph(g, flip=[0,0,0], switch=[0,0,0]):
    '''
    modify graph coordinates
    
    Input:
        
        m: the input mesh
        
        flip: parameters used to flip or not the coordinates: [x, y, z].
        
        switch: parameters used to switch or not coordinates: [xy, xz, yz]
    '''

    v=[g.node[i]['pos'] for i in g.GetNodes()]
    
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
    for idx, i in enumerate(g.GetNodes()):
        
        gg.node[i]['pos']=np.array([x[idx], y[idx], z[idx]])
    
    return gg


def calculateDistMap(label):

    shape=np.shape(label)
    
    XY=[label[i,:,:] for i in range(shape[0])] #Z-XY
    ZX=[label[:,:,i] for i in range(shape[2])] #Y-ZX 
    ZY=[label[:,i,:] for i in range(shape[1])] #X-ZY
    
    DistXY=np.array([image.morphology.distance_transform_edt(i) for i in XY])
    DistZX=np.array([image.morphology.distance_transform_edt(i) for i in ZX])
    DistZY=np.array([image.morphology.distance_transform_edt(i) for i in ZY])
    
    DistZX=np.rollaxis(DistZX, 0, 3)
    DistZY=np.rollaxis(DistZY, 0, 2)      
    
    DistMap_=np.maximum(DistXY, DistZX) 
    DistMap=np.maximum(DistMap_, DistZY)
    
    return DistMap


def  getMiddleGraph(G, thr):

    max_p1=0
    max_p2=0
    max_p3=0
    
    for i in G.GetNodes():
        
        if max_p1<G.node[i]['pos'][0]:
            max_p1=G.node[i]['pos'][0]
        
        if max_p2<G.node[i]['pos'][1]:
            max_p2=G.node[i]['pos'][1]
        
        if max_p3<G.node[i]['pos'][2]:
            max_p3=G.node[i]['pos'][2]

    
    
    for i in G.GetNodes():
        p1,p2,p3=G.node[i]['pos']       
        if p1>thr and p1<max_p1-thr and p2>thr and p2<max_p2-thr and p3>thr and p3<max_p3-thr:
            pass
        else:
            G.remove_node(i)

    return fixG(G)




def rescaleG(G, cube=512):
    
    p=np.array(G.GetNodesPos())
    pmin=np.min(p,axis=0)  
    pmax=np.max(p,axis=0) 
    
    for i in range(G.number_of_nodes()):
        try:
            p_=G.node[i]['pos']
            G.node[i]['pos']=512*(p_-pmin)/pmax
        except:
            p_=np.array(G.node[i]['pos'])
            G.node[i]['pos']=512*(p_-pmin)/pmax        
    return G


def getBranches(G):
    
    idNodes,_=findNodes(G)
    Gr=reduceG(G.copy())
    nbrs=[]    
    
    #find connections between nodes    
    for i in idNodes:
        nbrs_=Gr.GetNeighbors(i)
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


def getCoreGraph(G, indices):
    
    #obtain reduced graphs (with only burifications)
    Gr=reduceG(G.copy()) # note that Gr is not fixed by fixG
    
    nodesGr=Gr.GetNodes()
    
    nbrs=[]
    #find connections between nodes
    #connections=[]
    for i in Gr.GetNodes():
        nbrs_=Gr.GetNeighbors(i)
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
        
    for i in G.GetNodes():
        if len(G.GetNeighbors(i))<1:
            G.remove_node(i)
    
    return G



def LabelGraphBranchesOneSource(graph, source):
    
    from VascGraph.GeomGraph import GraphObject
     
    if graph.is_directed():
        gg=graph.copy().to_undirected()
    else:
        gg=graph.copy()
    
    if not nx.is_connected(gg.copy().to_undirected()):
        'This graph is not fully connected'
        return 
    
    obj=GraphObject(gg)
    obj.UpdateReducedGraph()
    obj.UpdateDictBranches()
    
    g=obj.GetReducedGraph()
    branches=obj.GetDictBranches()
    
    search=list(nx.bfs_edges(g, source)) # passes nodes
    g.node[search[0][0]]['branch']=1
    
    for i in search:
        
        node=i[0]
        next_node=i[1]
        
        try:
            is_there=g.node[next_node]['branch']
        except: 
            g.node[next_node]['branch']=g.node[node]['branch']+1
       
    
    
    for i in branches.keys():
        
        # get branch level from reduced graph
        n1, n2 = i
        b1, b2=[g.node[n1]['branch'], g.node[n2]['branch']]
        b=min(b1, b2)
        
        
        # set branch level on the origional graph for junction nodes
        gg.node[n1]['branch']=b1
        gg.node[n2]['branch']=b2
        
        # set branch level on the origional graph for rest nodes
        nodes=branches[i]
        nodes=[k1 for k2 in nodes for k1 in k2]
        for j in nodes:
            gg.node[j]['branch']=b
             
       
    b=[]  
    error=0
    nodes_error=[]
    
    for i in gg.GetNodes(): 
        try:
            b.append(gg.node[i]['branch'])
        except:
            error+=1
            nodes_error.append(i)
            pass
     
    # assign branch id to nodes in self loops   
    cont=0
    while cont==0:  
        cont=1
        for i in nodes_error:
            nbrs=gg.GetNeighbors(i)
            br=[]
            
            if len(nbrs)==0:
                print('This is not a fully connected graph!')
                break
            
            for j in nbrs:
                try:
                    br.append(gg.node[j]['branch'])
                except:
                    cont=0
                    pass
                
            if len(br)>0:
                br_=np.min(br)
                gg.node[i]['branch']=br_
    
    return [gg.node[i]['branch'] for i in gg.GetNodes()]


def LabelGraphBranchesManySources(graph, sources=[]):

    if type(sources)==int:
        sources=[sources]
    else:
        pass
    
    b=[]
    for s in sources:
        b.append(LabelGraphBranchesOneSource(graph, s))

    b=np.array(b)
    b_final=np.min(b, axis=0)

    for i,j in zip(b_final, graph.GetNodes()):
        graph.node[j]['branch']=i   



def TransferAttributes (DiG, G, warning=True):
     
    attr=['pos', 'r', 'type', 'branch', 'source', 'sink', 'root']
    
    for att in attr:
        for i in DiG.GetNodes():
            try:
                DiG.node[i][att]=G.node[i][att]
            except: pass
    
    edg=G.GetEdges()[0]
    attr_edg=G[edg[0]][edg[1]].keys()
    
    # attr that are not assigned to all edges
    rem=['outflow', 'inflow']
    for i in rem:
        try:
            attr_edg.remove(i)
        except:
            pass

    for att in attr_edg:
        try:
            for i in DiG.GetEdges():
                DiG[i[0]][i[1]][att]=G[i[0]][i[1]][att]
        except: 
            if warning:
                print('Warning: attribute \''+ att +'\' is not assigned to all graph edges!')
       
       
    ############## if attributes are set for some but not all of graph compartments 
    # ------------ edges ---------#
        
    for i in DiG.GetEdges():
        try:
            DiG[i[0]][i[1]]['inflow']=G[i[0]][i[1]]['inflow'] 
        except: pass
    
        try:
            DiG[i[0]][i[1]]['outflow']=G[i[0]][i[1]]['outflow'] 
        except: pass
    
        try:
            DiG[i[0]][i[1]]['pressure']=G[i[0]][i[1]]['pressure']  
        except: pass
    
    # --------- nodes ------------#
    for i in DiG.GetNodes():
        try:
            DiG.node[i]['inflow']=G.node[i]['inflow'] 
        except: pass
    
        try:
            DiG.node[i]['outflow']=G.node[i]['outflow'] 
        except: pass
    
        try:
            DiG.node[i]['sink']=G.node[i]['sink'] 
        except: pass
    
        try:
            DiG.node[i]['source']=G.node[i]['source'] 
        except: pass        
    
    

def TransferAttributesFaster(DiG, G, warning=True):
     
    attr=['pos', 'r', 'type', 'branch', 'res', 'vol', 'flow', 'pressure', 'label', 'node', 'type', 'area']
    
    # nodes and edges
    for att in attr:
        try:
            for i in DiG.GetNodes():
                DiG.node[i][att]=G.node[i][att]
        except: pass
        
        try:
            for i in DiG.GetEdges():
                DiG[i[0]][i[1]][att]=G[i[0]][i[1]][att]
        except: pass    
    
    
    # source  sink nodes
    nodes1,nodes2=G.GetSourcesSinks()
    nodes=nodes1
    nodes.extend(nodes2)
    

    attr=['source', 'sink', 'inflow', 'outflow', 'pressure']
    
    for i in nodes:
        for att in attr:
            try:
                DiG.node[i][att]=G.node[i][att]
            except: pass

    attr=['source', 'sink', 'root', 'pressure', 'inflow', 'outflow']
        
    for i in nodes:
        for att in attr:
            try:
                nn=DiG.GetNeighbors(i)
                if len(nn)>0:
                    for n in nn:
                        DiG[i][n][att]=G[i][n][att]
            except: pass

            try:
                nn=DiG.GetPredecessors(i)
                if len(nn)>0:
                    for n in nn:
                        DiG[n][i][att]=G[n][i][att]
            except: pass



