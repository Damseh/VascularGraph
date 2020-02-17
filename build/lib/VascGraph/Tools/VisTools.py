#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:05:59 2019

@author: rdamseh
"""
import matplotlib
import matplotlib.pyplot as plt

from mayavi import mlab

from mayavi.sources.api import ArraySource
from mayavi.filters.api import Stripper, Tube, WarpScalar, PolyDataNormals, Contour, UserDefined
from mayavi.filters.set_active_attribute import SetActiveAttribute
from mayavi.modules.api import IsoSurface, Surface, Outline, Text3D, Glyph
from mayavi.tools.pipeline import line_source 
from mayavi.core.api import Engine
from mayavi.tools.pipeline import vector_scatter, vectors

from VascGraph.Tools.CalcTools import *

def visGraph(p, e, d, r, radius=.15, color=(0.8,0.8,0), gylph_r=0.5, gylph_c=(1,1,1)):
    
    
    if d is not None:
        pts=mlab.points3d(p[:,0],p[:,1],p[:,2], d)
    elif r is not None:
        pts=mlab.points3d(p[:,0],p[:,1],p[:,2], r)
    else:
        pts=mlab.points3d(p[:,0],p[:,1],p[:,2], scale_factor=gylph_r, color=gylph_c)

    pts.mlab_source.dataset.lines = e 
    tube = mlab.pipeline.tube(pts, tube_radius=radius)
    tube.filter.set_input_data(pts.mlab_source.dataset)    
    surface=mlab.pipeline.surface(tube, color=color)

    if d is not None or r is not None:
        tube.filter.vary_radius = 'vary_radius_by_absolute_scalar'
        surface.actor.mapper.scalar_range = [ 0.0, np.max(d)]
        surface.actor.mapper.progress = 1.0
        surface.actor.mapper.scalar_visibility = True

def visG(G, 
         radius=.15, 
         color=(0.8,0.8,0), 
         gylph_r=0.5, 
         gylph_c=(1,1,1),
         diam=False,
         jnodes_r=None,
         jnodes_c=(0,0,1)):
    
    nodes=[G.node[i]['pos'] for i in G.GetNodes()]   
    edges=[[i[0],i[1]] for i in G.edges()]    
    
    #obtain radius in each node
    d=None
    r=None
    
    if diam:
        try:
            d=[G.node[i]['d'] for i in G.GetNodes()]   
        except:
            try:
                r=[G.node[i]['r'] for i in G.GetNodes()]
            except:
                pass
    
    nodes=np.array(nodes).astype('float') # importnat to be float type
    edges=np.array(edges)
    
    visGraph(p=nodes, e=edges, d=d, r=r,
             radius=radius,
             color=color, 
             gylph_r=gylph_r, 
             gylph_c=gylph_c)
    
    if jnodes_r:
        _, jnodes=findNodes(G, j_only=False)  
        x=[i[0] for i in jnodes]
        y=[i[1] for i in jnodes]
        z=[i[2] for i in jnodes]
        mlab.points3d(x,y,z,scale_factor=jnodes_r, color=jnodes_c)


def visStack(v, opacity=.5, color=(1,0,0), mode=''):
    
    if mode!='same':
        mlab.figure(bgcolor=(1,1,1)) 


    s=mlab.get_engine() # Returns the running mayavi engine.
    scene=s.current_scene # Returns the current scene.
    scene.scene.disable_render = True # for speed
    
    origion=[0,0,0]
    label= 'Segmentation'
    
    A=ArraySource(scalar_data=v)
    A.origin=np.array(origion)
    D=s.add_source(A)# add the data to the Mayavi engine
    #Apply gaussain 3d filter to smooth visualization
    
    
#    F=mlab.pipeline.user_defined(D, filter='ImageGaussianSmooth')
#    F.filter.set_standard_deviation(0,0,0)
    contour = Contour()
    s.add_filter(contour)
    
#    smooth = mlab.pipeline.user_defined(contour, filter='SmoothPolyDataFilter')
#    smooth.filter.number_of_iterations = 1
#    smooth.filter.relaxation_factor = 0  

    surface=Surface()
    s.add_module(surface)
    
    surface.module_manager.scalar_lut_manager.lut_mode = u'coolwarm'
    surface.module_manager.scalar_lut_manager.reverse_lut = True
    surface.actor.property.opacity = opacity
    surface.actor.mapper.scalar_visibility=False
    surface.actor.property.color = color #color
    
    return surface

def visVolume(v):
    src = mlab.pipeline.scalar_field(v)
    mlab.pipeline.volume(src)
    
def visVectors(x, y, z, u, v, w, legend=False):
    

        
    v_scatter=vector_scatter(x, y, z, u, v, w)
    vec=vectors(v_scatter)
    vec.actor.property.line_width = 2
    vec.glyph.glyph_source.glyph_source.scale = 2
    
    if legend:
        vec.module_manager.scalar_lut_manager.show_legend = True
        vec.module_manager.scalar_lut_manager.use_default_range=False
        vec.module_manager.scalar_lut_manager.use_default_name = False
        vec.module_manager.scalar_lut_manager.scalar_bar.title= 'Magnitude'
        vec.module_manager.scalar_lut_manager.number_of_labels = 0
        
    
    vec.glyph.glyph_source.glyph_source = vec.glyph.glyph_source.glyph_dict['arrow_source']
    vec.glyph.glyph_source.glyph_source.shaft_radius = 0.15
    vec.glyph.glyph_source.glyph_source.tip_length = 0.25
    vec.glyph.glyph_source.glyph_source.tip_radius = 0.35
    vec.module_manager.scalar_lut_manager.title_text_property.color = (0.0, 0.0, 0.0)

    
def createCam(position,
             focal_point,
             view_angle,
             view_up,
             clipping_range):
    
    cam=dict()
    
    cam['position']=position
    cam['focal_point']=focal_point    
    cam['view_angle']=view_angle
    cam['view_up']= view_up   
    cam['clipping_range']=clipping_range
    
    return cam
        
    
def setCam(cam=None):
    
    if cam:  
        
        e=mlab.get_engine()
        c=e.current_scene
        c.scene.camera.position = cam['position']
        c.scene.camera.focal_point = cam['focal_point']
        c.scene.camera.view_angle = cam['view_angle']
        c.scene.camera.view_up = cam['view_up']
        c.scene.camera.clipping_range = cam['clipping_range']
        c.scene.camera.compute_view_plane_normal()
        c.scene.render()    
      
    
