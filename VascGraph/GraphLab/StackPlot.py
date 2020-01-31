#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:16:06 2019

@author: rdamseh
"""

from mayavi.core.api import Engine, PipelineBase
from mayavi.tools.pipeline import scalar_field, surface, contour

from VascGraph.Tools.CalcTools import *
from mayavi import mlab 

class StackPlotParam:
    
    def __init__(self,
                SurfaceColor=(0.75, 0.0, 0.15),
                SurfaceOpacity=0.3):

        # visparam
        self.SurfaceColor=SurfaceColor
        self.SurfaceOpacity=SurfaceOpacity
               
        
class StackPlot:
    
    def __init__(self, s=None, param=None, new_engine=False):
        
        '''
        input: 
            param: object from StackPlotParam
        ''' 
        
        # if s is None:
        #     print('Noe input surface model')
        #     return
        
        
        if param is None:
            param=StackPlotParam()
        
        self.SurfaceColor=param.SurfaceColor
        self.SurfaceOpacity=param.SurfaceOpacity
        
        # start engine
        if new_engine:            
            from mayavi.core.api import Engine
            e=Engine()
            e.start()
            
        # source
        self.DataSource=scalar_field((np.random.rand(5,5,5)>0.5).astype(int))
        self.DataSource.origin=[0,0,0]
        self.Data=self.DataSource.scalar_data
        self.DataSource.origin=np.array([0,0,0])
        
        #modules
        self.Contour=contour(self.DataSource)
        self.Contour.filter.contours=[]
        self.Contour.filter.contours=[0.5]
        self.Contour.filter.auto_update_range = False
        
        self.Surface=None
        
        self.__UpdateSurface()
        
    def __UpdateSurface(self):
        
        if self.Surface is not None:
            self.Surface.remove()
            
        self.Surface=surface(self.Contour)
        self.Surface.actor.mapper.scalar_range = [ 0.0, 1.0] 
        self.Surface.actor.mapper.progress = 1.0
        self.Surface.actor.property.opacity = self.SurfaceOpacity
        self.Surface.actor.mapper.scalar_visibility = False
        self.Surface.actor.property.color = self.SurfaceColor 

    def Update(self, array):
        try:
            self.DataSource.scalar_data=array
            self.DataSource.update()
            self.Contour.update_data()
            self.__UpdateSurface()
            
        except:
            print('Cannot update stack plot!')

    def SetSurfaceColor(self, i):
        self.Surface.actor.mapper.scalar_visibility = False
        self.Surface.actor.property.color=i
     
    def SetSurfaceOpacity(self, i):
        self.Surface.actor.property.opacity = i
        
    def Remove(self):
        self.DataSource.remove()

