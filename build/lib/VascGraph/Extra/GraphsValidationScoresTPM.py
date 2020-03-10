#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:28:51 2018

@author: rdamseh
"""

from graphContraction import *
import pandas as pn

path='/home/rdamseh/IPMI paper/data/tpm/seg/'
pathgt='/home/rdamseh/IPMI paper/data/tpm/seg/'

if __name__=='__main__':
    
    scores={'meGFNR':[],'meshGFNR':[],'basicGFNR':[],
            'meGFPR':[],'meshGFPR':[],'basicGFPR':[],
            'meCFNR':[],'meshCFNR':[],'basicCFNR':[],
            'meCFPR':[],'meshCFPR':[],'basicCFPR':[]}
    sigma=[10]
    
    
    
    for i in [1]:
        
        # read graphs
        gtrue=readPAJEK(pathgt+'groundtruth/'+str(i)+'.pajek')
        gtrue=rescaleG(gtrue)
        
        # me
        gme=readPAJEK(path+'mygraphs/'+str(i)+'.pajek')
        gme=adjustGraph(gme, flip=[1,0,0], switch=[1,0,0])
        gme=rescaleG(gme)
        
        #mesh
        gmesh=readPAJEK(path+'meshgraphs/'+str(i)+'.pajek')
        gmesh=rescaleG(gmesh)

        # basic
        gbasic=readPAJEK(path+'basicgraphs/'+str(i)+'.pajek')
        gbasic=adjustGraph(gbasic, flip=[0,1,0], switch=[0,0,0])
        gbasic=rescaleG(gbasic)

        # validate
        validateme=validate(gtrue, gme, sigma=sigma, rescale=False, middle=32)
        validatemesh=validate(gtrue, gmesh, sigma=sigma, rescale=False, middle=32)
        validatebasic=validate(gtrue, gbasic, sigma=sigma, rescale=False, middle=32)
        
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


#scores=pn.DataFrame(scores)
#scores.to_csv(path+'scores_sigma'+str(sigma[0])+'.csv')








