#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:06:46 2018

@author: rdamseh
"""
from graphContraction import *

path='/home/rdamseh/IPMI paper/data/sim/data16noisy1/'
#path='/home/rdamseh/IPMI paper/data/tpm/noisy/seg/'

for i in [1,2,3,4,5,6,7,8,9,10]:
   
    v=path+'meshgraphs/vertices'+str(i)+'.cgal'
    e=path+'meshgraphs/edges'+str(i)+'.cgal'
    g=getGraphfromCGAL(v, e)
    g=fixG(g)
    
    nx.write_pajek(g,path+'meshgraphs/'+str(i)+'.pajek')
    g=readPAJEK(path+'meshgraphs/'+str(i)+'.pajek')
    visG(g)

        
        