#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:24:25 2019

@author: rdamseh
"""


from mayavi.core.api import Engine, PipelineBase

from mayavi.tools.pipeline import scalar_scatter, glyph, tube, surface
from VascGraph.Tools.CalcTools import *
from mayavi import mlab 
import scipy as sp


class GraphPlotParam:
    
    def __init__(self,
                 TubeColor=(0.75, 0.75, 0.15),
                 TubeRadius=0.5,
                 GylphColor=(0.75, 0.15, 0.15),
                 GylphSize=1.0):

        self.TubeColor=TubeColor
        self.TubeRadius=TubeRadius
        self.GylphColor=GylphColor
        self.GylphSize=GylphSize

        

class GraphPlot():
    
    def __init__(self, new_engine=False, param=None):
        
        '''
        Input:
            param: object of GraphPlotParam class
        '''   
        if param is None:
            param=GraphPlotParam()
        
        self.TubeColor=param.TubeColor
        self.TubeRadius=param.TubeRadius
        self.GylphColor=param.GylphColor
        self.GylphSize=param.GylphSize
        
        
        self.TubeMode={'radius_by_scale':False, 
                       'radius_by_color':False, 
                       'type_by_color':False,
                       'branching_by_color':False,
                       'flow_by_color':False,
                       'pressure_by_color': False}  
        
        # start engine
        if new_engine:            
            from mayavi.core.api import Engine
            e=Engine()
            e.start() 
            
        # source
        self.DataSource=scalar_scatter([],[],[])
        self.Data=self.DataSource.data
        self.Nodes=self.Data.points
        self.Edges=self.Data.lines
        
        #modules
        self.Glyph=glyph(self.DataSource)
        self.Glyph.glyph.glyph.clamping = True
        self.Glyph.glyph.glyph.scale_factor = 2.0
        self.Glyph.glyph.glyph.range = np.array([0., 1.])
        self.Glyph.glyph.color_mode = 'no_coloring'
        self.Glyph.glyph.scale_mode = 'data_scaling_off'
        self.Glyph.actor.property.representation = 'wireframe'
        self.Glyph.actor.property.color=(0.75,0.25,0.25)
        self.Tube=tube(self.Glyph, tube_radius=self.TubeRadius)   
        self.TubeSurface=surface(self.Tube, color=self.TubeColor)
        self.Tube.filter.vary_radius = 'vary_radius_off'
        self.TubeSurface.actor.mapper.scalar_visibility = False

    def __UpdatePlot(self): 
        
        #nodes
        if self.Nodes is not None:
            points=np.array([i for i in self.Pos])
            self.Nodes.reset()
            self.Nodes.from_array(points)
            self.DataSource.update()
 
        #edges
        if self.Connections is not None:
            edges=np.array([i for i in self.Connections])
            self.Edges.reset()
            self.Data.lines=edges
            self.DataSource.update()
            self.Tube.filter.set_input_data(self.Data) 
           
        self.TubeSurface.actor.mapper.scalar_range = [ 0.0, 10.0] 
        self.TubeSurface.actor.mapper.progress = 1.0
        self.AssigneTubePointsToGlyphPoints()
        
        if self.TubeMode['radius_by_scale']:
            self.SetTubeRadiusByScale(i=True)
       
        if self.TubeMode['radius_by_color']:
            self.SetTubeRadiusByColor(i=True)
       
        if self.TubeMode['type_by_color']:
            self.SetTubeTypeByColor(i=True)
     
        if self.TubeMode['branching_by_color']:
            self.SetTubeBranchingByColor(i=True)   

        if self.TubeMode['flow_by_color']:
            self.SetTubeFlowByColor(i=True) 
            
            
            
    def __GetInfofromGraph(self, graph):
        
        self.Graph=graph
        
        # graph info
        try:
            self.Pos=self.Graph.NodesPosIter
            self.Connections=self.Graph.EdgesIter
        except:
            self.Pos=None
            self.Edges=None
            print('No geometric graph found!') 
            
        try:
            self.Radii=self.Graph.RadiiIter
        except:
            self.Radii=None
            print('No radius assigned to graph nodes!')         
       
        try:
            self.Types=self.Graph.TypesIter
        except:
            self.Types=None
            print('No types assigned to graph nodes!')     
        try:
            self.BranchLabels=self.Graph.BranchLabelsIter
        except:
            self.BranchLabels=None
            print('No branch labels assigned to graph nodes!')   
            
        
            
    def Update(self, graph):
        self.__GetInfofromGraph(graph)
        self.__UpdatePlot()

    def GetGlyphSourcePoints(self):
        return self.Glyph.glyph.glyph_source.glyph_source.output.points.to_array()
    
    def GetGlyphPoints(self):
        return self.Data.points.to_array()
    
    def GetTubeData(self):
        self.Tube.update_data()
        return self.Tube.get_output_dataset()
    
    def GetTubePoints(self):
        return self.GetTubeData().points.to_array()
  
    def GetTubePointData(self):
        return self.GetTubeData().point_data

    def GetTubePointDataScalars(self):
        return self.GetTubeData().point_data.scalars.to_array()

    def AssigneTubePointsToGlyphPoints(self):
        Tree = sp.spatial.cKDTree(self.GetGlyphPoints())
        self.IndTubePointsToGlyphPoints=Tree.query(self.GetTubePoints(), k=1)[1]   

    def SetTubeColor(self, i):
        self.TubeSurface.actor.mapper.scalar_visibility = False
        self.TubeSurface.actor.property.color=i
        
    def SetTubeRadius(self, i): 
        self.Tube.filter.radius=i
        
    def SetGylphColor(self, i):
        self.Glyph.actor.property.color=i

    def SetGylphSize(self, i): 
        self.Glyph.glyph.glyph.scale_factor = i

    def SetTubeRadiusByScale(self, i=None):
  
        if i is True and self.Radii is not None:
          
            self.Data.point_data.scalars=self.Graph.GetRadii()
            self.Tube.filter.vary_radius = 'vary_radius_by_scalar'
            self.TubeMode['radius_by_scale']=True
        else:
            self.Tube.filter.vary_radius = 'vary_radius_off'
            self.TubeMode['radius_by_scale']=False

    def SetTubeRadiusByColor(self, i=None):
        if i is True and self.Radii is not None:
          
            GlyphScalars=self.Graph.GetRadii()
            self.TubeScalars=[GlyphScalars[i] for i in self.IndTubePointsToGlyphPoints]
            self.GetTubePointData().scalars=self.TubeScalars
            self.Tube.update_data()
          
            self.TubeSurface.actor.mapper.scalar_visibility = True
            self.TubeMode['radius_by_color']=True

        else:
            self.TubeSurface.actor.mapper.scalar_visibility = False
            self.TubeMode['radius_by_color']=False

    def SetTubeTypeByColor(self, i=None):
        if i is True and self.Types is not None:
           
            GlyphScalars=self.Graph.GetTypes()
            self.TubeScalars=[GlyphScalars[i] for i in self.IndTubePointsToGlyphPoints]
            self.GetTubePointData().scalars=self.TubeScalars
            self.Tube.update_data()
          
            self.TubeSurface.actor.mapper.scalar_visibility = True
            self.TubeMode['type_by_color']=True

        else:
            self.TubeSurface.actor.mapper.scalar_visibility = False
            self.TubeMode['type_by_color']=False
    
    def SetTubeFlowByColor(self, i=None):
        if i is True and self.Types is not None:
           
            GlyphScalars=self.Graph.GetFlows()
            self.TubeScalars=[GlyphScalars[i] for i in self.IndTubePointsToGlyphPoints]
            self.GetTubePointData().scalars=self.TubeScalars
            self.Tube.update_data()
          
            self.TubeSurface.actor.mapper.scalar_visibility = True
            self.TubeMode['flow_by_color']=True

        else:
            self.TubeSurface.actor.mapper.scalar_visibility = False
            self.TubeMode['flow_by_color']=False

    def SetTubePressureByColor(self, i=None):
        if i is True and self.Types is not None:
           
            GlyphScalars=self.Graph.GetPressures()
            self.TubeScalars=[GlyphScalars[i] for i in self.IndTubePointsToGlyphPoints]
            self.GetTubePointData().scalars=self.TubeScalars
            self.Tube.update_data()
          
            self.TubeSurface.actor.mapper.scalar_visibility = True
            self.TubeMode['pressure_by_color']=True

        else:
            self.TubeSurface.actor.mapper.scalar_visibility = False
            self.TubeMode['flow_by_color']=False 

    def SetTubeVelocityByColor(self, i=None):
        if i is True and self.Types is not None:
           
            GlyphScalars=self.Graph.GetVelocities()
            self.TubeScalars=[GlyphScalars[i] for i in self.IndTubePointsToGlyphPoints]
            self.GetTubePointData().scalars=self.TubeScalars
            self.Tube.update_data()
          
            self.TubeSurface.actor.mapper.scalar_visibility = True
            self.TubeMode['pressure_by_color']=True

        else:
            self.TubeSurface.actor.mapper.scalar_visibility = False
            self.TubeMode['flow_by_color']=False
           
    def SetTubeBranchingByColor(self, i=None):
        
        if i is True and self.BranchLabels is not None:
            
            GlyphScalars=self.Graph.GetBranchLabels()
            self.TubeScalars=[GlyphScalars[i] for i in self.IndTubePointsToGlyphPoints]
            self.GetTubePointData().scalars=self.TubeScalars
            self.Tube.update_data()
            
            self.TubeSurface.actor.mapper.scalar_visibility = True
            self.TubeMode['branching_by_color']=True

        else:
            self.TubeSurface.actor.mapper.scalar_visibility = False
            self.TubeMode['branching_by_color']=False        
        
    def Remove(self):
        self.DataSource.remove()
















        