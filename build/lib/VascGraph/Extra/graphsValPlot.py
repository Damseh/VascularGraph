#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 15:46:04 2018

@author: rdamseh
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:27:18 2018

@author: rdamseh
"""

from util_validation import *
import scipy as sc
from matplotlib import pyplot as plt



def plotc(data, s, linestyle='-', marker='8', color=(0,0,0), label=None, linewidth=1):
        
        if label:            
            plt.plot(s, data, label=label, 
                         marker=marker,
                         color=color,
                         linestyle=linestyle,
                         markevery=0.05, 
                         linewidth=linewidth)
        else:

            plt.plot(s, data,
                 marker=marker,
                 color=color,
                 linestyle=linestyle,
                 markevery=0.05, 
                 linewidth=linewidth)
                



if __name__=='__main__':


    path='graphs/'
    names1=['gr1', 'gr2', 'gr3', 'gr4', 'gr5', 'gr6']    
    names2=['g1', 'g2', 'g3', 'g4', 'g5', 'g6']        
    names3=['gbasic1', 'gbasic2', 'gbasic3', 'gbasic4', 'gbasic5', 'gbasic6']        


    graphs1=[rescaleG(readPAJEK(path+i+'.pajek')) for i in names1]
    graphs2=[rescaleG(readPAJEK(path+i+'.pajek')) for i in names2]
    
    graphs33=[rescaleG(readPAJEK(path+i+'.pajek')) for i in names3]
    graphs3=[getFullyConnected(i) for i in graphs33 ]
    graphs3=[adjustGraph(i, flip=[0,1,0]) for i in graphs3]


    savepath='graphs/figures/'



##########
### metrics
##########

    # create validation objects + get scores
    valsbasic_middle=[validate(i, j, sigma=[10,20,30,40,50,60], rescale=True, middle=64)
                for i, j in zip(graphs1, graphs3)]   
    sbasic_middle=[i.scores() for i in valsbasic_middle]
    
    valsbasic_full=[validate(i, j, sigma=[10,20,30,40,50,60], rescale=True, middle=False)
                for i, j in zip(graphs1, graphs3)]   
    sbasic_full=[i.scores() for i in valsbasic_full]


####
# plotting
#####


    s=[10,20,30,40,50,60]
    markers=['^','o','s','*','x','d']
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors=colors[:6]
    labels=['$Dataset$'+str(n) for n in [1,2,3,4,5,6]]
    

    gFNRmiddle=[sbasic_middle[i][0] for i in [0,1,2,3,4,5]]
    gFPRmiddle=[sbasic_middle[i][1] for i in [0,1,2,3,4,5]]
    cFNRmiddle=[sbasic_middle[i][2] for i in [0,1,2,3,4,5]]
    cFPRmiddle=[sbasic_middle[i][3] for i in [0,1,2,3,4,5]]

    gFNRfull=[sbasic_full[i][0] for i in [0,1,2,3,4,5]]
    gFPRfull=[sbasic_full[i][1] for i in [0,1,2,3,4,5]]
    cFNRfull=[sbasic_full[i][2] for i in [0,1,2,3,4,5]]
    cFPRfull=[sbasic_full[i][3] for i in [0,1,2,3,4,5]]


    mets, names= [[gFNRfull, gFNRmiddle],
                 [gFPRfull, gFPRmiddle],
                 [cFNRfull, cFNRmiddle],
                 [cFPRfull, cFPRmiddle]], ['GFNR','GFPR','CFNR','CFPR']

    
    bgcolor=(0.8584083044982699, 0.9134486735870818, 0.9645674740484429)
    
    for met, name in zip(mets, names):        
        
        met1, met2 = met[0], met[1]
        
        f=plt.figure(figsize=(8,6), facecolor=(1,1,1))
        
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
            
        ax = f.add_subplot(1, 1, 1) # nrows, ncols, index
        ax.set_facecolor(bgcolor)
        
        plt.hold
        plt.grid(color=(1,1,1))
        

        
        for n, (i, j) in  enumerate(zip(met1, met2)):

            
            plotc(s=s, data=i, linestyle='-', 
                  marker=markers[n], 
                  color=colors[n], 
                  label=labels[n], 
                  linewidth=2)
            
            plotc(s=s, data=j, linestyle='-.', 
                  marker=markers[n], 
                  color=colors[n], 
                  label=None)
               
    
        plt.legend(fontsize=18, frameon=True, facecolor=(1,1,1)) 
        plt.ylabel(name ,fontsize = 20); plt.xlabel('$\delta$' ,fontsize = 20) 
        plt.xticks(s, fontsize = 16)
        plt.yticks(np.arange(0, 1+.1, 0.1), fontsize = 16)   
        
        plt.savefig(savepath+name+'_basic'+'.eps', format='eps', dpi=1000, transparent=True)
    
    
    
    
    
    