#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:25:10 2018

@author: rdamseh
"""


from util_ import *
from util_validation import *
import scipy.spatial as sp

from sklearn.cluster import DBSCAN
import sklearn as sk


def connect_surg(G, nodes_to_merg, cen_nodes_to_merg, cen_med_val):
    """
    This funtion modify the topolgy of the graph 'G'.    
    
    Input:
        "G": The graph to be modified. 
        
        "nodes_to_merg": Indicies of Graph nodes that will be replaced 
        by one node. The elements in each row of 'nodes_to_merg' represent the indices of 
        a group of nodes in the Graph 'G' that will be  contracted into ine node.
        
        "cen_nodes_to_merg": Geometric postions [x,y,z] of the nodes to be added to the graph 'G'.
        Each row in 'cen_nodes_to_merg' represent the position of one node that will replace the 
        nodes with indices in the corresponding row of 'nodes_to_merg'.
    
    Output:
        "G": Modified graph.
            
    """
    nbrs_of_nbrs=[]
    new_l=[]
    connect_to_nbrs=[]
    n_v=G.number_of_nodes()
    
    for i in tqdm(range(len(cen_nodes_to_merg))):
     
        # get neighbours of veticies belonging to this cluster 
        nbrs_=nodes_to_merg[i]
        nbrs_=list(set(G.nodes()).intersection(set(nbrs_)))
        
        # add new vertix of this cluster centroid 
        new_l_=n_v+i
        new_l.append(new_l_)
        G.add_node(new_l_)
        G.node[new_l_]['pos']=np.array(cen_nodes_to_merg[i])
        G.node[new_l_]['node']=False # tage the node with False
                    #assign new med_val
        G.node[new_l_]['med_val']=cen_med_val[i] 
        
        nbrs_of_nbrs_=[G.neighbors(nbrs_[j]) for j in range(len(nbrs_))]
        
        #flatten the list: "nbrs_of_nbrs_"
        nbrs_of_nbrs_= [nbrs_of_nbrs_[m][n] for m in range(len(nbrs_of_nbrs_)) for n in range(len(nbrs_of_nbrs_[m]))] 
        
        #get unique indices
        nbrs_of_nbrs_=list(set(nbrs_of_nbrs_))
        
        # retain only neighbours of neighbours that are still in the graph
        nbrs_of_nbrs_=list(set(G.nodes()).intersection(set(nbrs_of_nbrs_)))
     
        # append the neighbor of neighbours for this vertix
        nbrs_of_nbrs.append(nbrs_of_nbrs_)
        
        #build new connections
        connect_to_nbrs=[  [new_l_, nbrs_of_nbrs_[k]] for k in range(len(nbrs_of_nbrs_))]
        G.add_edges_from(connect_to_nbrs) 
        
        #return edges that are still in the graph
        G.remove_nodes_from(nbrs_)
        
    return G


def DBSCAN_cluster(X, eps=2):
    """
    Cluster a group of points based on their geometric postions.
    
    Input:
        "x", "y", "z": The location of the point in the space at 
        the x, y, z coordinates.
        
        "eps": Sensitivity of DBSCAN clustering, in our algorithm it is fixed to 2.
        
    Output:
        "centroids": Geometric position for the centers of the clusters as [x,y,z].
        "points": The Geometric position for the points groubed in each cluster.
        "poin_ind":The indecies of points gropued in each cluster.   
        
    """       
    db = DBSCAN(eps=eps, min_samples=1).fit(X)
    labels = db.labels_    
    unique_labels = list(set(labels))
        
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_= len(unique_labels) - (1 if -1 in labels else 0)
        
    poin_ind = [np.where(labels == i)[0]
       for i in unique_labels]    
    poin=[X[i] for i in poin_ind]
    centroids=[np.mean(i,axis=0) for i in poin]
        
    return centroids, poin, poin_ind

def AssignToClusters(pos):
    '''
    Assigne the current nodes in graph with closest pixel
    
    Input:
        pos: position of the current graph nodes
        
    Output:
        centroids: Geometric position for the centers of the clusters as [x,y,z].
        clusters_pos: qeometric positin of the nodes grouped in eache cluster.
        clusters_points: The indecies of points gropued in each cluster.   
        
        '''
        
    clusters_init=np.round(pos).astype(int)
    
    c, clusters_index, clusters_inverse, clusters_count=np.unique(clusters_init, 
                             axis=0, return_inverse=True, return_index=True, 
                             return_counts=True)  
    
    clusters=np.where(clusters_count>1)[0]
    clusters_points=[np.where(clusters_inverse==i)[0] for i in clusters]
    
    clusters_pos=[pos[i] for i in clusters_points]
    centroids=[np.mean(i,axis=0) for i in clusters_pos]

    return centroids, clusters_pos, clusters_points


def createGraphFromSeg(label, n_points=1000000, connect=16):
    """
    
    """
    #### random sampling of the grid #### 
    s=np.shape(label) # size of image

    #create random points
    x=np.random.uniform(low=0, high=s[0], size=n_points).tolist()
    y=np.random.uniform(low=0, high=s[1], size=n_points).tolist()
    z=np.random.uniform(low=0, high=s[2], size=n_points).tolist()
       
    
    #### build graph from sampled grid ####
    G=nx.Graph()
    G.add_nodes_from(range(n_points))
    
    # assign coordinates 
    val=np.Inf    
    for i in tqdm(range(G.number_of_nodes())):
        G.node[i]['pos']=np.array([x[i],y[i],z[i]]) 
        
    #get a segmented graph
    G_seg=G.copy()
    for i in tqdm(range(G.number_of_nodes())):
        n_pos = G_seg.node[i]['pos']
        if label[int(n_pos[0]), int(n_pos[1]), int(n_pos[2])] == 0:
            G_seg.remove_node(i)
    G_seg=fixG(G_seg)    



    #### build connectivity ####
    n_nodes=G_seg.number_of_nodes()
    nodes=[[G_seg.node[i]['pos'][0],
                    G_seg.node[i]['pos'][1],
                    G_seg.node[i]['pos'][2]] for i in range(n_nodes)]


    #construct connections 
    tree = sp.cKDTree(nodes)
    connections=dict()
    for i in tqdm(range(int(n_nodes))):
        indDist=tree.query(nodes[i], k=connect, distance_upper_bound=100)[1]
        indDist=np.unique(indDist)
        connections[str(i)]=[[i, indDist[j]]  for j in range(len(indDist))
                             if indDist[j] != i and indDist[j] != n_nodes]
     
    #assign connections    
    for i in connections.keys():
        if connections[str(i)]:
            G_seg.add_edges_from(connections[str(i)])  
           
    G_seg=G_seg.to_undirected() 
    
    #Get connected components   
    sizes=[]
    graphs=list(nx.connected_component_subgraphs(G_seg))
    for i in range(len(graphs)):
        s=graphs[i].size()
        sizes.append(s)
        

    sizes=np.array(sizes)
    ind=np.where(sizes==np.max(sizes))
    ind=np.array(ind).ravel()
    ind=ind[0]
    G_seg=graphs[ind]
    G_seg=fixG(G_seg)  
    
    return G_seg



def createGraphFromSeg2(label, n_points=250000, connect=8, portion=None):
    """
    
    """
    
    print('Create random nodes on segmentation')
    
    #### random sampling of the grid ####        
    index=np.array(np.where(label>0)).T
    
    nvoxels=len(index)
    if portion:
        n_points=int(nvoxels*portion)
    
#    ######################################    
#     #un comment to sample at subvoxel level
#    level=2
#    for i in range(level):
#        v=np.zeros(len(index)).astype('int8')
#        l=1/2**(i+1)
#        ind1=index+np.array([v+l, v, v]).T
#        ind2=index+np.array([v-l, v, v]).T
#        ind3=index+np.array([v, v+l, v]).T
#        ind4=index+np.array([v, v-l, v]).T
#        ind5=index+np.array([v, v, v+l]).T
#        ind6=index+np.array([v, v, v-l]).T 
#        index=np.vstack((index, ind1, ind2, ind3, ind4, ind5, ind6))
#        ##################################
#    
    
    index=sk.utils.shuffle(index)
    
    #### build graph from sampled grid ####
    G=nx.Graph()
    G.add_nodes_from(range(n_points))
    
    # assign coordinates     
    nodes=index[:n_points]
    n_nodes=G.number_of_nodes()    
    for idx, i in tqdm(enumerate(nodes)):
        G.node[idx]['pos']=i
        
 
    #### build connectivity ####

    #construct connections 
    print('Biuld KDTree . . .')
    tree = sp.cKDTree(nodes)
    indDist=tree.query(nodes, k=connect, distance_upper_bound=10)[1]
    
    print('Assign connections to graph . . .')
    c=[]
    for idx, i in enumerate(indDist):
        indD=np.unique(i)
        cc=[[idx, j]  for j in indD
                             if j != idx and j != n_nodes]
        if cc:
            c.append(cc)
    
    #assign connections         
    connections=[j for i in c for j in i] 
    G.add_edges_from(connections)  
                
    return G


def createGraphFromSeg3(label, n_points=200000):
    """
    
    """ 
    #### random sampling of the grid #### 
    ind=np.where(label) # 
    
    #limit n_points to the number of labeld pixels
    if n_points>np.shape(ind)[1]:
        n_points=np.shape(ind)[1]
        
    p=(label).astype(float)/np.sum(label)
    p=p[ind]
    indnodes=np.random.choice(range(len(p)), n_points, p=p)
    ind=np.array(ind).T
    nodes=ind[indnodes]
    
    def Edistance(n1, n2):
        dist=(n1.astype('float')-n2)**2
        return np.sqrt(np.sum(dist,1))
    
    ###################   
    # obtain delauny triangulation.
    dl=sc.spatial.Delaunay(nodes, qhull_options='QJ')
    c=dl.vertices
    c=np.vstack((c[:,(0,1)], c[:,(0,2)], c[:,(0,3)],
                   c[:,(1,2)], c[:,(1,3)], c[:,(2,3)]))

    ###################   
    #refine delauny 1##
    # line equations for connections
    x0y0z0=nodes[(c[:,0])].astype('float')
    x1y1z1=nodes[(c[:,1])]
    abc=x1y1z1-x0y0z0 
    t=np.arange(.1,1,.1)
    # check if lines coress nonlabeled area
    xyz=np.array([x0y0z0+i*abc for i in t]) 
    xyz=np.floor(xyz).astype(int)
    check=np.array([label[(xyz[i,:,0],xyz[i,:,1],xyz[i,:,2])] 
            for i in range(len(t))])
    not_edge=set(np.where(check==0)[1])
    is_edge=set(range(len(c))).difference(not_edge)
    is_edge=tuple(is_edge)
    c=c[is_edge, :] 

    ###################   
    #refine delauny 2
    p1=nodes[(c[:,0])].astype('float')
    p2=nodes[(c[:,1])]    
    dist=Edistance(p1,p2)
    distscore=np.array(sc.stats.zscore(dist)) # remove outlieres 
    #exclude outier edges
    maxd=np.max(dist[distscore<3]) 
    is_edge=np.where(dist<=maxd)[0]
    c=c[is_edge, :] 

    ###################   
    #build graph   
    vertices=list(set(np.ravel(c)))
    nodes=nodes[vertices]
    g=nx.Graph()
    g.add_nodes_from(vertices)
    for idx, i in enumerate(g.nodes()):
        g.node[i]['pos']=nodes[idx]
    g.add_edges_from(c)
    g=getFullyConnected(g)    
    return g




def createGraphFromSeg4(label):
    
    
    def pixelPos(s):
        
        '''
        s: shape of array
        
        indexing in order: rows by row->depth
        '''
            # positions of each pixel
        pos = np.meshgrid(range(s[1]), range(s[2]),range(s[0]))
        pos = np.transpose([np.ravel(pos[2], 'F'), 
                            np.ravel(pos[0], 'F'), 
                            np.ravel(pos[1], 'F') ])
    
        length=s[0]*s[1]*s[2]
        n = np.reshape(range(length), s) 

        return n, pos
    
    
    s=label.shape
    length=s[0]*s[1]*s[2]
    #s=(2,3,4)
        
    # pixel indices and thier positions
    n, pos=pixelPos(s)
      
    # pathways in pixels to create connections
    c1=np.ravel(n) 
    c2=np.ravel(np.transpose(n, axes=(0,2,1))) 
    c3=np.ravel(n, 'F') 

    c1, c2, c3 = c1.reshape((length/s[2],s[2])),\
        c2.reshape((length/s[1],s[1])),\
            c3.reshape((length/s[0],s[0]))
  
    c11=[np.ravel(c1[:, 0:-1]), np.ravel(c1[:, 1:])]
    c22=[np.ravel(c2[:, 0:-1]), np.ravel(c2[:, 1:])]
    c33=[np.ravel(c3[:, 0:-1]), np.ravel(c3[:, 1:])]

    c11=np.transpose(c11)
    c22=np.transpose(c22)
    c33=np.transpose(c33)

    c=np.vstack((c11, c22, c33))

    # add nodes
    nodes=np.ravel(n)
    G=nx.Graph()
    G.add_nodes_from(nodes)

    # add connectiones
    G.add_edges_from(c)

    
    # false nodes
    nFalse=n[label==0]

    # true nodes
    nTrue=n[label>0]
    posTrue=pos[nTrue]

    # assign pos
    for i, j in zip(nTrue, posTrue):
        G.node[i]['pos']=j
    
    G.remove_nodes_from(nFalse)


    false_n=[i for i in G.nodes() if len(G.neighbors(i))<3]
    G.remove_nodes_from(false_n)
    G=fixG(G)
   
    G=getFullyConnected(G)
    
    return fixG(G)



def createGraphFromSeg5(label, sample=2):

    def pixelPos(s):
        
        '''
        s: shape of array
        
        indexing in order: rows by row->depth
        '''
            # positions of each pixel
        pos = np.meshgrid(range(s[1]), range(s[2]),range(s[0]))
        pos = np.transpose([np.ravel(pos[2], 'F'), 
                            np.ravel(pos[0], 'F'), 
                            np.ravel(pos[1], 'F') ])
    
        length=s[0]*s[1]*s[2]
        n = np.reshape(range(length), s) 

        return n, pos
    
    sample=float(sample)
    scale=(1/sample, 1/sample, 1/sample)    
    label=ndim.zoom(label, scale)
    
    s=np.shape(label)
    length=s[0]*s[1]*s[2]
    
    # pixel indices and thier positions
    n, pos=pixelPos(s)
  
    # pathways in pixels to create connections
    c1=np.ravel(n) 
    c2=np.ravel(np.transpose(n, axes=(0,2,1))) 
    c3=np.ravel(n, 'F') 

    c1, c2, c3 = c1.reshape((length/s[2],s[2])),\
        c2.reshape((length/s[1],s[1])),\
            c3.reshape((length/s[0],s[0]))
            
    c11=[np.ravel(c1[:, 0:-1]), np.ravel(c1[:, 1:])]
    c22=[np.ravel(c2[:, 0:-1]), np.ravel(c2[:, 1:])]
    c33=[np.ravel(c3[:, 0:-1]), np.ravel(c3[:, 1:])]

    c11=np.transpose(c11)
    c22=np.transpose(c22)
    c33=np.transpose(c33)
    c=np.vstack((c11, c22, c33))

############# Check valid connections ##################  
    chck1=np.ravel(label>0) 
    chck2=np.ravel(np.transpose(label>0, axes=(0,2,1)))
    chck3=np.ravel(label>0, 'F') 
    
    chck1, chck2, chck3 = chck1.reshape((length/s[2],s[2])),\
        chck2.reshape((length/s[1],s[1])),\
            chck3.reshape((length/s[0],s[0]))

    chck1=[np.ravel(chck1[:, 0:-1]), np.ravel(chck1[:, 1:])]
    chck2=[np.ravel(chck2[:, 0:-1]), np.ravel(chck2[:, 1:])]
    chck3=[np.ravel(chck3[:, 0:-1]), np.ravel(chck3[:, 1:])]    

    chck1=np.transpose(chck1)
    chck2=np.transpose(chck2)
    chck3=np.transpose(chck3)
    chck=np.vstack((chck1, chck2, chck3)) 
    chck=np.all(chck, axis=1)
    
#######################################################        

    c=c[chck]
    
    # false nodes
    nFalse=n[label==0]
    
    # true nodes
    nTrue=n[label>0]
    posTrue=pos[nTrue]*sample

    # add nodes
    G=nx.Graph()
    G.add_nodes_from(nTrue)
   
    # assign pos
    for i, j in zip(nTrue, posTrue):
        G.node[i]['pos']=j

    # add connectiones
    G.add_edges_from(c)

    false_n=[i for i in G.nodes() if len(G.neighbors(i))==2]
    G.remove_nodes_from(false_n)
    
    false_n=[i for i in G.nodes() if len(G.neighbors(i))==1]
    G.remove_nodes_from(false_n)

    G=getFullyConnected(G)
    G=fixG(G)
    
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

def cycleAreaAll(corners):
        
    if len(corners):       
        n = np.shape(corners)[1] # of corners
        cross= np.zeros((np.shape(corners)[0],np.shape(corners)[2]))
        for i in range(n):
            j = (i + 1) % n
            crss=np.cross(corners[:,i],corners[:,j])
            cross=cross+crss
            #nrm+=np.linalg.norm(crss)
        area = np.linalg.norm(cross, axis=1) / 2.0
    else:
        return 0
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



def checkNode(a, b, thr=0):
    
    '''
    A fucntion to check a certian node is 
    to be processec in next iteration. The condoction is based on angles 
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
    
    mask=norm2>0
    cosine_angle =  np.ones_like(norm2)   
    
    if norm1>0:
        cosine_angle[mask]=dot[mask]/(norm1*norm2[mask])    
    
    mask=cosine_angle<0
    cosine_angle[mask]=cosine_angle[mask]+1
    
    add=np.zeros_like(norm2)  
    add[mask]=np.pi/2
    
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle+add)
     
    thr1, thr2 = thr, 180-thr
    chck=(angle>thr1)&(angle<thr2)
    
    return not np.any(chck) #true if skel


def isSklNodes(p, pn, thr=0):
          
    '''
    output a boolian array with True values incidicating skeletal nodes
    '''
    
    # check nodes to process
    chck=[checkNode(i,j, thr) for i, j in zip(p, pn)]
   
    return np.array(chck) # true if  skeleton



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





def isSklNodes2(g):
      
            
#    '''
#    NOT WORKING
#    output a boolian array with True values incidicating skeletal nodes
#    '''
#    thr_area=.1
#    ln=10
#    cyc=nx.cycle_basis(g) 
#    
#    
#    areas=[]
#    nodes=[]  
#    for l in range(ln):
#        if l>2:
#            
#            t=[k for k in cyc if len(k)==l]              
#            #positon of poly vertices
#            p=np.array([[g.node[j]['pos'] for j in i]  for i in t])
#            # get polys that pass the area condition        
#            areas.append(cycleAreaAll(p))
    pass



def GetSilancyMap(im):
    
    pass














