#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:07:34 2018

@author: rdamseh
"""

from graphContraction import *


if __name__=='__main__':
    

    path='data/sim/data56/'
    
    for i in [1,2,3,4,5,6,7,8,9,10]:
        
        filename=path+'t'+str(i)+'.mat'
    
        t=tree(filename=filename)
        G=t.getG()
        nx.write_pajek(G, path+'/groundtruth/'+str(i)+'.pajek')            
        
        
        
        