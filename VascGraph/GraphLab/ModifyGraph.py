#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:58:09 2018

@author: rdamseh
"""



from VascGraph.Tools.VisTools import *
from VascGraph.GraphIO import ReadPajek

from pyface.api import FileDialog, OK
from numpy import arange, pi, cos, sin
from traits.api import HasTraits, Range, Instance, Enum, \
        on_trait_change, Button, String, Float, Array, List
from traitsui.api import View, UItem, Item, Group, TitleEditor, ListEditor

from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import MayaviScene, SceneEditor, \
                MlabSceneModel
import time

import numpy as np

class ModifyGraph(HasTraits):

    #visualization items
    scene = Instance(MlabSceneModel, ())
    plot = Instance(PipelineBase)
  
    #control items
    Glyph_size=Range(low=.1, high=5.0, value=1)
    Tube_radius=Range(low=.1, high=5.0, value=1)
    Bound_size=Range(low=.1, high=5.0, value=1)
    Forground_color=Enum('Black','White','Red', 'Green', 'Blue', 'Yellow')
    Glyph_color=Enum('Black','White','Red', 'Green', 'Blue', 'Yellow')
    Tube_color=Enum('Black','White','Red', 'Green', 'Blue', 'Yellow')
    
    #edit items
    current_node=String()
    nodes_list=List(editor=ListEditor())
    reset_nodes=Button(label='Reset')
    connect_nodes=Button(label='Add branch')
    remove_node=Button(label='Remove')
    save=Button(label='Save graph')

    #others
    colors={'Black':(0,0,0),'White':(1,1,1),'Red':(1,0,0), 
            'Green':(0,1,0), 'Blue':(0,0,1), 'Yellow':(1,1,0)}
    

    def __init__(self, G_, **traits):
        
        HasTraits.__init__(self, **traits)
        
        # Visualization
        self.G_=G_
        self.G=None
        self.engine=None
        self.scn=None
        self.pipeline=None
        self.tube=None
        self.surface=None
        self.glyph=None
        self.glyph_points=None # points needed to render one glyph
        self.outline=None
        self.x, self.y, self.z= None, None, None
        self.node_id=None
        self.data=None
        self.nodes=None
        self.edges=None
        
        #parameters
        self.bound=3
        self.bending_factor=40
        self.connect_step=.1
        self.n_new_nodes=0
    
    
    ##################
    #Fuctions on scene
    ##################
    
    def clearScene(self):
        
        mlab.clf()
        mlab.close(self.scene)
        
     
    def update_selection(self):
        self.current_node=self.node_id
        self.nodes_list=self.nodes_list.append(self.node_id)
        
    def update_picker_opt(self):
        
        b=self.bound
        self.outline.outline_mode = 'full'
        self.outline.bounds = (self.x-b, self.x+b,
                               self.y-b, self.y+b,
                               self.z-b, self.z+b)             
    def update_picker(self):
        
        if self.x and self.y and self.z:
            pass
        else:
            self.x, self.y, self.z= self.nodes[0] 
         
        b=self.bound
        
        if self.outline:            
            self.outline.bounds = (self.x-b, self.x+b,
                                   self.y-b, self.y+b,
                                   self.z-b, self.z+b) 
        else:
                        
            self.outline = mlab.outline(line_width=3, color=(0,0,0))
            self.outline.outline_mode = 'full'
            self.outline.bounds = (self.x-b, self.x+b,
                                   self.y-b, self.y+b,
                                   self.z-b, self.z+b) 
        self.update_selection()
    
    
    def update_data(self):
         
        self.nodes.reset()
        all_pos=[self.G.node[i]['pos'] for i in self.G.GetNodes()]       
        self.nodes.from_array(all_pos)               
        
        self.edges.reset()
        self.edges.from_array(np.array(self.G.edges()))               
        
        self.pipeline.update()
     
    ####################    
    # Functions on graph
    ####################  
    
    def get_bezier(self, pnts):
                
        
        try:
            import bezier as bz
        except:
            print('To run this function, \'bezier\' sould be installed.')
            return 
        
        #to well-bended curve
        v1=(pnts[1]-pnts[0])/np.linalg.norm(pnts[1]-pnts[0])
        v2=(pnts[2]-pnts[3])/np.linalg.norm(pnts[2]-pnts[3])   
        pnts[1]=pnts[1]+self.bending_factor*v1
        pnts[2]=pnts[2]+self.bending_factor*v2
        
        x=pnts[:,0].tolist(); y=pnts[:,1].tolist(); z=pnts[:,2].tolist(); 
           
        nodes = np.asfortranarray([x,y,z])
        curve = bz.Curve(nodes, degree=2)    
        stp=self.connect_step
        steps=np.arange(0+stp, 1-stp, stp)
        
        new_pnts=[]
        for i in steps:    
            new_pnts.append(np.ravel(curve.evaluate(i)))
        
        return np.array(new_pnts)    
    
    def add_branch4(self):
        
        #new_nodes
        pos=np.array([
                self.G.node[i]['pos'] 
                for i in list(self.nodes_list)
                ]) 

        new_pos=self.get_bezier(pos)
               
        self.n_new_nodes=np.shape(new_pos)[0]
        new_nodes=range(len(self.nodes), 
                      len(self.nodes)+ self.n_new_nodes) 
        
        #new_connections
        srt=self.nodes_list[1]
        end=self.nodes_list[-2]
        new_con=[[new_nodes[i], new_nodes[i+1]] 
                for i in range(len(new_nodes)-1)]
        new_con.append([srt,new_nodes[0]])
        new_con.append([new_nodes[-1],end]) 
        
        #add nodes
        self.G.add_nodes_from(new_nodes)
        
        for idx, i in enumerate(new_nodes):
            self.G.node[i]['pos']=new_pos[idx]
        
        #add connectinos
        self.G.add_edges_from(new_con)
      
       
        #update visualization
        self.update_data()
        
    
    def add_branch2(self):
    
        #add connectinos
        self.G.add_edge(self.nodes_list[0],
                               self.nodes_list[1])
            
        #update visualization
        self.update_data()    
   
    def add_branch(self):         
        
        if len(self.nodes_list)==4:           
            self.add_branch4()
            
        if len(self.nodes_list)==2:           
            self.add_branch2()
            
        
    
    def rm_node(self):
        
        self.G.remove_node(self.node_id)
        self.G=fixG(self.G)
        self.update_data()
        
    def init_graph(self):
        
        if not self.G:
            self.G=self.G_.copy()
        
  
    
    # control call backes    
    def _Glyph_size_changed(self): 
        
       self.glyph.glyph.glyph_source.glyph_source.radius = self.Glyph_size
       
    # control call backes    
    def _Bound_size_changed(self): 
        
       self.bound = self.Bound_size
       self.update_picker_opt()
       
    def _Glyph_color_changed(self): 
        
       self.glyph.actor.property.color=self.colors[self.Glyph_color]
           
    def _Tube_radius_changed(self): 
        
       self.tube.filter.radius=self.Tube_radius
                 
    def _Tube_color_changed(self): 
        
       self.surface.actor.property.color=self.colors[self.Tube_color]
       
       
       
    # edit call backs    
    def _reset_nodes_fired(self):
        self.nodes_list=[]


    def _connect_nodes_fired(self):
        
        self.add_branch()
        
    def _save_fired(self):
    
        
        dlg = FileDialog(action='save as')
        if dlg.open() == OK:
            if dlg.filename:
                #os.mkdir(dlg.filename)
                nx.write_pajek(self.G,
                               dlg.directory+'/'+dlg.filename+'.pajek')

    def _remove_node_fired(self):
        self.rm_node()
    
    @on_trait_change('scene.activated')
    def update_plot(self):
        
        if self.plot is None:                        
            visG(self.G_, radius=.1, color=(0,0,1), gylph_r=1, gylph_c=(0,1,0))
            self.init_graph()
            self.scene.scene.background = (1.0, 1.0, 1.0)
            self.engine=mlab.get_engine()
            self.scn=self.engine.scenes[1]
            self.pipeline=self.scn.children[0]
            self.tube=self.scn.children[0].children[1]
            self.surface=self.scn.children[0].children[1].children[0].children[0]
            self.glyph=self.scn.children[0].children[0].children[0]
            self.glyph_points = self.glyph.glyph.glyph_source.glyph_source.output.points.to_array()
            self.data=self.glyph.mlab_source.dataset
            self.nodes=self.data.points
            self.edges=self.data.lines
            
            self.figure=mlab.gcf(engine=self.engine)
            
            def picker_callback(picker):
                """ Picker callback: this get called when on pick events.
                """
                if picker.actor in self.glyph.actor.actors:
        
                    self.node_id = picker.point_id/self.glyph_points.shape[0]
                    
                    if self.node_id != -1:
                        self.x, self.y, self.z = self.nodes[self.node_id]
                        self.update_picker()
            
            self.picker = self.figure.on_mouse_pick(picker_callback)             
            self.picker.tolerance = 0.01
            
            self.update_picker()
        else:

            pass
    
    
    ##############
    # control tool
    ##############
    
    control_group= Group(
                       Group(
                           Item('Glyph_size'), 
                           Item('Glyph_color'),
                           orientation='horizontal',
                           layout='normal'
                           ),
                   
                       Group(
                           Item('Tube_radius'), 
                           Item('Tube_color'), 
                           orientation='horizontal',
                           layout='normal'
                           ),
                               
                       Group(
                           Item('Bound_size'), 
                           orientation='horizontal',
                           layout='normal'
                           ),                              
                               
                    label='Control', layout='normal')
   
    ###########
    # edit tool
    ###########
  
    editing_group=   Group(
                     
                     Group(
                     Item('current_node', label='Current node', editor=TitleEditor()),
                     Item('remove_node', show_label=False)),
                     
                     Group(
                             Item('reset_nodes', show_label=False),
                             Item('nodes_list', style='readonly', label='Selected nodes'),
                             Item('connect_nodes', show_label=False),
                             orientation='horizontal'
                            ),
                             
                     Group('_', Item('save', show_label=False),'_'),
                     
                     label='Edit', orientation='vertical', layout='split',show_border = True,
                     )

    # The layout of the dialog created
    view = View(
            Group(
                
               Group(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=600, width=400, show_label=False)),
             
               Group(control_group,
               editing_group, layout='tabbed'),
                     
                layout='split'),
               resizable=True,    title = 'Nodes selection'    
               
               )


if __name__=='__main__':
    g=ReadPajek('/home/rdamseh/GraphPaper2018V1/1.pajek')
    window=ModifyGraph(fixG(g.GetOutput()))
    window.configure_traits()