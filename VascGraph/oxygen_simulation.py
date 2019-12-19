#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 09:45:36 2019

@author: rdamseh
"""

import sys
sys.path.append('/home/rdamseh/GraphPaper2018V1')

import scipy as sp
import scipy.io as sio
from scipy.optimize import curve_fit
import numpy as np
import networkx as nx
import skimage as sk
from VascGraph.Skeletonize import GenerateGraph, ContractGraph, RefineGraph
from VascGraph.GeomGraph import DiGraph
from VascGraph.GraphIO import ReadPajek, WritePajek
from VascGraph.GraphLab import GraphPlot
from matplotlib import pyplot as plt
from mayavi import mlab
from VascGraph.Tools.VisTools import *
from VascGraph.Tools.CalcTools import *
from GraphFlowSimulation import GraphObject, GraphFlowSimulation

if __name__=="__main__":


    g=ReadPajek('/home/rdamseh/GraphPaper2018V1'+'/flow/test_network_flow.di.pajek').GetOutput()
    
    arteriol_pressure=60
    venule_pressure=25
    intra_cranial_pressure=10
    
    input_so2=.94 # arteriol o2 saturation
    alpha= 1.275*10e-12 # bunsen solubility
    po2_tissue=15 # o2 partial pressure in tissue
    
    rho=30*1e-6 # effective radius for o2 consumbtion rate
    Kappa_w= 5*10e-8 #o2 premiability on vascular wall
    C_Hb= 150  # hemoglobin concentration g/mL
    gamma_Hb=2.3*10-9 # mL of Hb to moles




    def so2_to_Cb(so2): 
        
        Hb=1 # mass of hemoglobin in grams
        Cb=1.36*Hb*so2 # bounded o2 concentrartion 
        
        return Cb
    
    
    def po2_Cf(po2):
        Cf=0.0031*po2 # Concetration of free o2 (dissolved)
        return Cf





    

    # po2-so2 funtion
    def po2_so2(po2, model='mouse'):
        '''
        Converts the partial pressure of oxygen into o2 saturation
        based on disassociation curve
        '''

        # assumptions : pco2=40; temp= 37 
        
        if model=='mouse':
            
            # C57BL/6 mice, Uchida K. et al., Zoological Science 15, 703-706, 1997
            
            n=2.59 # hill curve
            ph=7.4
            p50=40.2 # for single Beta-globin type
        
      
        elif model=='rat':
            # Cartheuser, Comp Biochem Physiol Comp Physiol, 1993
            
            n=2.6
            ph=7.4
            p50=36

            
        po2s=po2*(10**(0.61*(ph-7.4)))
        so2=po2s**n/(po2s**n+p50**n)        
        return so2
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    po2=np.linspace(5.0, 100.0, 1000)  
    so2=PO2ToSO2(po2)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
