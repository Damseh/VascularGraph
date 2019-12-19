#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:28:51 2018

@author: rdamseh
"""

from graphContraction import *
import pandas as pn

path='/home/rdamseh/IPMI paper/data/sim/data16noisy1/'
pathgt='/home/rdamseh/IPMI paper/data/sim/data16/'


if __name__=='__main__':
    
    scores={'meGFNR':[],'meshGFNR':[],'basicGFNR':[],
            'meGFPR':[],'meshGFPR':[],'basicGFPR':[],
            'meCFNR':[],'meshCFNR':[],'basicCFNR':[],
            'meCFPR':[],'meshCFPR':[],'basicCFPR':[]}
    sigma=[10]
    
    
    
    for i in [1,2,3,4,5,6,7,8,9,10]:
        
        # read graphs
        gtrue=readPAJEK(pathgt+'groundtruth/'+str(i)+'.pajek')
        gtrue=adjustGraph(gtrue, flip=[0,0,0], switch=[0,1,0])

        # me
        gme=readPAJEK(path+'mygraphs/'+str(i)+'.pajek')
        gme=reduceG(gme)
        
        #mesh
        gmesh=readPAJEK(path+'meshgraphs/'+str(i)+'.pajek')
        gmesh=adjustGraph(gmesh.copy(), flip=(0,1,0), switch=(0,1,0)) 
        gmesh=reduceG(gmesh)

        gbasic=readPAJEK(path+'basicgraphs/'+str(i)+'.pajek')
        gbasic=adjustGraph(gbasic, flip=[0,0,0], switch=[0,1,0])
        
        # validate
        validateme=validate(gtrue, gme, sigma=sigma, rescale=False)
        validatemesh=validate(gtrue, gmesh, sigma=sigma, rescale=False)
        validatebasic=validate(gtrue, gbasic, sigma=sigma, rescale=False)
        
        s=validateme.scores()
        scores['meGFNR'].append(s[0][0])
        scores['meGFPR'].append(s[1][0])
        scores['meCFNR'].append(s[2][0])
        scores['meCFPR'].append(s[3][0])

        s=validatemesh.scores()
        scores['meshGFNR'].append(s[0][0])
        scores['meshGFPR'].append(s[1][0])
        scores['meshCFNR'].append(s[2][0])
        scores['meshCFPR'].append(s[3][0])
        
        s=validatebasic.scores()
        scores['basicGFNR'].append(s[0][0])
        scores['basicGFPR'].append(s[1][0])
        scores['basicCFNR'].append(s[2][0])
        scores['basicCFPR'].append(s[3][0])


scores=pn.DataFrame(scores)
scores.to_csv(path+'scores_sigma'+str(sigma[0])+'.csv')
