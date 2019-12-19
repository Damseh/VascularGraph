#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:44:48 2019

@author: rdamseh
"""

import os
import sys
sys.path.append(os.getcwd())

from VascGraph.GraphLab import  MainDialogue, ModifyGraph
from VascGraph.GraphIO import ReadPajek, WritePajek,  ReadSWC


if __name__=='__main__':
    
    # ------------ #
    window=MainDialogue()
#    idx=1
#    name='soaxgraph'
#    g=ReadPajek('validation/radius/synth/56noisy2/'+name+str(idx)+'.pajek').GetOutput() 
#    window=ModifyGraph(g)
#    window.configure_traits()
