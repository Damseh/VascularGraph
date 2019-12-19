#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 11:19:07 2019

@author: rdamseh
"""

import os
import sys
try:
    sys.path.append('/home/rdamseh/GraphPaper2018V1')
except: pass

from VascGraph.GraphLab import GraphPlot, StackPlot
from VascGraph.GraphIO import ReadPajek, ReadStackMat, ReadSWC, WriteSWC, WritePajek
from VascGraph.GeomGraph import Graph 
from VascGraph.GeomGraph import AnnotateDiGraph
from VascGraph.Tools.CalcTools import *
from VascGraph.Skeletonize import GenerateGraph, ContractGraph, RefineGraph, RefineGraphRadius

from pyface.api import FileDialog, OK
from traits.api import Trait, HasTraits, Range, Instance, Enum, \
        on_trait_change, Button, String, Float, Array, List, Bool, Int, ListInt
from traitsui.api import View, UItem, Item, Group, TitleEditor, ListEditor, ImageEnumEditor, TextEditor


from mayavi.core.api import Engine, PipelineBase
from mayavi.core.ui.api import MayaviScene, SceneEditor, \
                MlabSceneModel
from mayavi import mlab

import scipy.io as sio
import numpy as np
from time import time

class MainDialogue(HasTraits):
    
    '''
    class of main interface
    '''

    #visualization items ##########################
    MyEngine=Engine()
    MyEngine.start()
    Scene = Instance(MlabSceneModel(engine=MyEngine), ())
    Status1=String()
    Status2=String()

    #control items ##########################
    Surface_color=Enum('Black','White','Red', 'Green', 'Blue', 'Yellow')
    Surface_opacity=Range(low=0, high=1.0, value=0.2)
    Load_stack=Button(label='Load stack')
    Remove_stack=Button(label='Remove stack')

    Gylph_size=Range(low=.01, high=5.0, value=2)
    Tube_radius=Range(low=.01, high=5.0, value=1)
    Selection_size=Range(low=.01, high=5.0, value=2)
    Forground_color=Enum('Black','White','Red', 'Green', 'Blue', 'Yellow')
    Gylph_color=Enum('Black','White','Red', 'Green', 'Blue', 'Yellow')
    Tube_color=Enum('Black','White','Red', 'Green', 'Blue', 'Yellow')
    TubeRadiusByScale=Bool(label='Radius by scale')
    TubeRadiusByColor=Bool(label='Radius by color')
    TubeTypeByColor=Bool(label='Type by color')
    TubeBranchingByColor=Bool(label='Branching by color')

    TubeFlowByColor=Bool(label='Flow by color')
    TubeVelocityByColor=Bool(label='Velocity by color')
    TubePressureByColor=Bool(label='Pressure by color')
    

    Load_graph=Button(label='Load graph')
    Save_graph=Button(label='Save graph')
    MetaData=Bool(label='MetaData')
    
    Save_type=Enum('pajek', 'swc')
    Remove_graph=Button(label='Remove graph')
    
    Load_camera=Button(label='Load camera')
    Save_camera=Button(label='Save camera')
    
    Save_snapshot=Button(label='Save_snapshot')
    Snapshot_resolution=Enum('1','2', '3', '4', '5')


    #skeleton items ##########################
    Skeletonize=Button(label='Skeletonize')
    Sampling=Range(low=0.5, high=3, value=1.5)
    SpeedParam=Range(low=0.001, high=1, value=0.05)
    DistParam=Range(low=0.001, high=1, value=0.01)
    MedParam=Range(low=0.001, high=1, value=1)
    SetDefault=Button(label='SetDefault')
    
    Initial_graph=Enum('Grid','Random')
    InitNodes=String(value='None')
    Connections=String(value='None')
    
    StopParam=String(label='Stopping Param')
    FreeIter=String(label='Number of Free Iter')
    Angle=String(label='Angle')
    ClusteringRes=String(label='Clustering resolution')

    PolyArea=String(label='Min Poly Area')
    PolyNum=String(label='Min Number of Poly')
    
    StopParam=0.005
    FreeIter=10
    Angle=5.0
    ClusteringRes=1.0
    PolyArea=50.0
    PolyNum=10
    

    #edit items ##########################
    GraphFilePath=String()
    StackFilePath=String()
    NodeID=String()
    NodeType=String()
    NodeBranch=String()
    NodeRadius=String()
    NodePos=String()
    nodes_list=List(editor=ListEditor())
    reset_nodes=Button(label='Reset')
    connect_nodes=Button(label='Add branch')
    Remove_node=Button(label='Remove node')

    # Labeling items ##########################
    Degree_cutoff=Range(low=3, high=100, value=3)
    Smooth_radius=Button(label='Smooth radius')
    Smoothing_mode=Enum('mean', 'median', 'max', 'min')
    Fix_radius=Button(label='Fix radius on branches')
    Fixing_mode=Enum('mean', 'median', 'max', 'min')
    
    Add_source=Button(label='Add source')
    Remove_source=Button(label='Remove source')
    Add_sink=Button(label='Add sink')
    Remove_sink=Button(label='Remove sink')
    Labeling_reset=Button(label='Reset')
    Sources=ListInt(maxlen=10, items=False)
    Sinks= ListInt(maxlen=10, items=False)
    Generate_directed_graph=Button(label='Generate directed graph')
    Generate_and_label_directed_graph=Button(label='Generate-Label directed graph')
    Prune_directed_graph=Button(label='Prune directed graph')
    Recall_undirected_graph=Button(label='Recall undirected graph')
    Label_branches=Button(label='Label branches')
    Label_tree_branches=Button(label='Label branches')
    
    Convert_to_tree_graph=Button(label='Convert to tree graph')
    Forest=Bool(label='Forest')

    icons_digraph=['true','false']
    path=os.path.dirname(os.path.realpath(__file__))
    
    DiGraph_check = Trait(editor=ImageEnumEditor(values=icons_digraph,
                                                   path=path),*icons_digraph)
    PrunedDiGraph_check = Trait(editor=ImageEnumEditor(values=icons_digraph,
                                                   path=path),*icons_digraph)
    Tree_check = Trait(editor=ImageEnumEditor(values=icons_digraph,
                                                   path=path),*icons_digraph)
    
    Propagate_vessel_type=Button(label='Propagate vessel type')
    Propagate_cutoff=Int()
    Vessel_type=Enum('Artery','Vein', 'Capillary')


    
    # flow items ##########################
    Compute_resistances=Button(label='Compute_resistances')
    Resistances_check = Trait(editor=ImageEnumEditor(values=icons_digraph,
                                                   path=path),*icons_digraph)    
    Compute_flow=Button(label='Compute flow')
    Flow_check = Trait(editor=ImageEnumEditor(values=icons_digraph,
                                                   path=path),*icons_digraph)   
    
    Generate_directed_graph_f=Button(label='Generate directed graph')
    
    Generate_directed_graph_f=Button(label='Generate directed graph')
    Prune_directed_graph_f=Button(label='Prune directed graph')
    Recall_undirected_graph_f=Button(label='Recall undirected graph')
    Label_branches_f=Button(label='Label branches')    
    
    DiGraph_check_f = Trait(editor=ImageEnumEditor(values=icons_digraph,
                                                   path=path),*icons_digraph)    
    Propagate_vessel_type_f=Button(label='Propagate vessel type')
    Propagate_cutoff_f=Int()
    Vessel_type_f=Enum('Artery','Vein', 'Capillary')


    #others ##########################
    TypeValue={'Artery': 10, 'Vein':20, 'Capillary':30}
    ValueType={0:'No Label!', 1: 'Artery', 2: 'Vein', 3:'Capillary'}
    colors={'Black':(0,0,0),'White':(1,1,1),'Red':(1,0,0), 
            'Green':(0,1,0), 'Blue':(0,0,1), 'Yellow':(1,1,0)}
        
    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)

        self.MyScene=self.Scene.mayavi_scene.scene
        self.MyScene.background=(.8, .9, .8)
        self.Figure=self.MyScene.mayavi_scene


        self.Graph=None
        self.DiGraphObject=AnnotateDiGraph(self.Graph)
        self.GraphPlot=None
        self.StackPlot=None
        self.MetaData=True
        self.DiGraph_check='false'
        self.PrunedDiGraph_check='false'
        self.Tree_check='false'
        self.DiGraph_check_f='false'
        self.Flow_check='false'
        self.Resistances_check='false'
        self.Snapshot_resolution='3'


        self.__Run()
        

# -------------------------------------------------------
# Traits notification
# -------------------------------------------------------
     
    def _SetDefault_fired(self):
        self.Sampling= 1
        self.SpeedParam= 0.05
        self.DistParam= 0.5
        self.MedParam= 1
        self.StopParam=0.001
        self.FreeIter=5
        self.Angle=5.0
        self.PolyArea=75.0
        self.PolyNum=10
         
    def _Load_graph_fired(self):
        self.GraphFilePath='' 
        dlg = FileDialog(action='open', title="Load graph")
        if dlg.open() == OK:
            if dlg.filename:
                self.GraphFilePath=dlg.directory+'/'+dlg.filename   
    
    def _Load_stack_fired(self):
        self.StackFilePath=''
        dlg = FileDialog(action='open', title="Load stack")
        if dlg.open() == OK:
            if dlg.filename:
                self.StackFilePath=dlg.directory+'/'+dlg.filename  

    def _GraphFilePath_changed(self):
        if self.GraphFilePath !='':
            self.__UpdateGraph()
            self.__UpdateGraphPlot()
            
 
    def _StackFilePath_changed(self):
        if self.StackFilePath !='':
            self.__UpdateStack()
            self.__UpdateStackPlot()    
    
    def _Save_graph_fired(self):
        dlg = FileDialog(action='save as', title="Save graph")
        if dlg.open() == OK:
            if dlg.filename:

                if self.Save_type=='swc':
                    
                    #if tree 
                    if not nx.is_tree(self.Graph):
                        print('Must be a tree graph to save in this format!')
                        return
                    
                    if '.' not in dlg.filename:
                        WriteSWC(dlg.directory+'/', dlg.filename+'.swc', self.Graph, self.DiGraphObject.TreeRoot)
                    elif dlg.filename.split('.')[-1]=='swc':
                        WriteSWC(dlg.directory+'/', dlg.filename, self.Graph, self.DiGraphObject.TreeRoot)
                    else:
                        WriteSWC(dlg.directory+'/', 
                                 ''.join(dlg.filename.split('.')[:-1])+'.swc', 
                                 self.Graph, self.DiGraphObject.TreeRoot)
                     
              
                if self.Save_type=='pajek':
                    
                    if self.Graph.is_directed():
                        #if directd
                        if '.' not in dlg.filename:
                            WritePajek(dlg.directory+'/', dlg.filename+'.di.pajek', self.Graph)
                        elif dlg.filename.split('.')[-2]=='di' and dlg.filename.split('.')[-1]=='pajek':
                            WritePajek(dlg.directory+'/', dlg.filename, self.Graph) 
                        else:
                            WritePajek(dlg.directory+'/', 
                                       ''.join(dlg.filename.split('.')[:-1])+'.di.pajek', 
                                       self.Graph)  
                    else:
                        # if undircted
                        if '.' not in dlg.filename:
                            WritePajek(dlg.directory+'/', dlg.filename+'.pajek', self.Graph)
                        elif dlg.filename.split('.')[-1]=='pajek':
                            WritePajek(dlg.directory+'/', dlg.filename, self.Graph) 
                        else:
                            WritePajek(dlg.directory+'/' , 
                                       ''.join(dlg.filename.split('.')[:-1])+'.pajek', 
                                       self.Graph)   
                            
                if self.MetaData:
                    dic={'sinks': np.array(self.Sinks), 'sources': np.array(self.Sources)}
                    sio.savemat(dlg.directory+'/'+dlg.filename+'.metadata.mat', dic)
    
        
    def _Save_camera_fired(self):
        
        dlg = FileDialog(action='save as', title="Save cam")
        if dlg.open() == OK:
            if dlg.filename:
                path=dlg.directory+'/'
                name=dlg.filename
                
                if '.' not in dlg.filename:
                    name=name+'.cam'
                elif name.split('.')[-1]=='cam':
                    pass
                else:
                    name=name+'.cam'
                    
                self.__SaveCamParam(path, name)
       


    def _Load_camera_fired(self):
        
        dlg = FileDialog(action='open', title="Load cam")
        if dlg.open() == OK:
            if dlg.filename:
                self.__LoadCamParam(dlg.directory+'/', dlg.filename)         
                
    def _Remove_graph_fired(self):
        self.__ResetGraphAll()
        
    def _Remove_stack_fired(self):
        self.Stack=None
        self.__RemoveStackPlot()

    def _Save_snapshot_fired(self):
        
        dlg = FileDialog(action='save as', title="Save snapshot")
        if dlg.open() == OK:
            if dlg.filename:
                path=dlg.directory+'/'
                name=dlg.filename
                
                if '.' not in dlg.filename:
                    name=name+'.png'
                elif name.split('.')[-1]=='png':
                    pass
                else:
                    name=name+'.png'
                    
                    
        res=float(self.Snapshot_resolution)
        size=tuple(np.array(self.MyScene.get_size())*res)  
        
        mlab.savefig(path+name, size=size)   
    
    def _Remove_node_fired(self):
        if self.NodeID is not None:
            self.Graph.remove_node(int(self.NodeID))
            self.Graph=fixG(self.Graph)
            self.DiGraphObject.SetGraph(self.Graph)
            self.GraphPlot.Update(self.Graph)
        
    def _Skeletonize_fired(self):
        if self.Stack is not None:
            self.__UpdateSkeleton()
    
    def _Initial_graph_changed(self):
        
        if self.Initial_graph=='Random':
            self.InitNodes=10000
            self.Connections=8
        else:
            self.InitNodes='None'
            self.Connections='None'
            
            
    def _Smooth_radius_fired(self):
        if self.Graph is not None:
            self.__UpdateRadius()
     
    def _Fix_radius_fired(self):
        if self.Graph is not None:
            self.__UpdateRadius(fixed=True)
            
    def _Surface_color_changed(self):  
       self.StackPlot.SetSurfaceColor(self.colors[self.Surface_color])

    def _Surface_opacity_changed(self):  
       self.StackPlot.SetSurfaceOpacity(self.Surface_opacity)
          
    def _Gylph_size_changed(self):  
       self.GraphPlot.SetGylphSize(self.Gylph_size)
 
    def _Gylph_color_changed(self): 
       self.GraphPlot.SetGylphColor(self.colors[self.Gylph_color])
    
    def _Tube_radius_changed(self): 
       self.GraphPlot.SetTubeRadius(self.Tube_radius)

    def _Tube_color_changed(self): 
       self.GraphPlot.SetTubeColor(self.colors[self.Tube_color])

    def _TubeRadiusByScale_changed(self):
       self.GraphPlot.SetTubeRadiusByScale(self.TubeRadiusByScale)
       
    def _TubeRadiusByColor_changed(self): 
       self.TubeTypeByColor=False
       self.TubeBranchingByColor=False
       self.TubeFlowByColor=False
       self.TubeVelocityByColor=False
       self.TubePressureByColor=False
       self.GraphPlot.SetTubeRadiusByColor(self.TubeRadiusByColor)
    
    def _TubeTypeByColor_changed(self): 
       self.TubeRadiusByColor=False
       self.TubeBranchingByColor=False
       self.TubeFlowByColor=False
       self.TubeVelocityByColor=False
       self.TubePressureByColor=False
       self.GraphPlot.SetTubeTypeByColor(self.TubeTypeByColor)

    def _TubeFlowByColor_changed(self): 
       self.TubeRadiusByColor=False
       self.TubeBranchingByColor=False
       self.TubeTypeByColor=False
       self.TubeVelocityByColor=False
       self.TubePressureByColor=False
       self.GraphPlot.SetTubeFlowByColor(self.TubeFlowByColor)

    def _TubePressureByColor_changed(self): 
       self.TubeRadiusByColor=False
       self.TubeBranchingByColor=False
       self.TubeTypeByColor=False
       self.TubeVelocityByColor=False
       self.TubeFlowByColor=False       
       self.GraphPlot.SetTubePressureByColor(self.TubePressureByColor)

    def _TubeVelocityByColor_changed(self): 
       self.TubeRadiusByColor=False
       self.TubeBranchingByColor=False
       self.TubeTypeByColor=False
       self.TubeFlowByColor=False
       self.TubePressureByColor=False
       self.GraphPlot.SetTubeVelocityByColor(self.TubeVelocityByColor)
       
    def _TubeBranchingByColor_changed(self): 
       self.TubeRadiusByColor=False
       self.TubeFlowByColor=False
       self.TubePressureByColor=False       
       self.TubeTypeByColor=False
       self.TubeVelocityByColor=False
       self.GraphPlot.SetTubeBranchingByColor(self.TubeBranchingByColor)
       
    def _Selection_size_changed(self): 
        self.UpdateNodeOutline()

    def _Add_source_fired(self):
        if int(self.NodeID) not in self.Sources and int(self.NodeID) not in self.Sinks: 
            self.Sources.append(int(self.NodeID))
            
    def _Add_sink_fired(self): 
        if int(self.NodeID) not in self.Sinks and int(self.NodeID) not in self.Sources: 
            self.Sinks.append(int(self.NodeID))
            
    def _Remove_source_fired(self): 
        if int(self.NodeID) in self.Sources: 
            self.Sources.remove(int(self.NodeID))
            
    def _Remove_sink_fired(self): 
        if int(self.NodeID) in self.Sinks: 
            self.Sinks.remove(int(self.NodeID))
            
    def _Labeling_reset_fired(self):
        self.Sources=[]
        self.Sinks=[]
     
    def _Generate_directed_graph_fired(self):
        
        if self.Graph.is_directed():
            self.Graph=self.Graph.to_undirected()
            self.DiGraph_check='false'
            
        self.DiGraphObject.SetGraph(self.Graph)
        
        if len(self.Sources)==0:
            print('Sources need to be set!')
        else:
            self.DiGraphObject.UpdateDiGraphFromGraph(Sources=self.Sources, Sinks=self.Sinks)
            self.DiGraph_check='true'
            
        self.Graph=fixG(self.DiGraphObject.GetDiGraph())
        self.__UpdateSourcesSinks()
        self.__UpdateGraphPlot()
        
        
    def _Generate_and_label_directed_graph_fired(self):
        
        if self.Graph.is_directed():
            self.Graph=self.Graph.to_undirected()
            self.DiGraph_check='false'
            
        self.DiGraphObject.SetGraph(self.Graph)
        
        if len(self.Sources)==0:
            print('Sources need to be set!')
        else:
            self.DiGraphObject.UpdateDiGraphFromGraph2(Sources=self.Sources, Sinks=self.Sinks)
            self.DiGraph_check='true'
            
        self.Graph=fixG(self.DiGraphObject.GetDiGraph())
        self.__UpdateSourcesSinks()
        self.__UpdateGraphPlot()
        
        
    def _Recall_undirected_graph_fired(self):
        
        if self.Graph.is_directed():
            pass
        else:
            print('Directed graph is needed!')
            return
        
        self.Graph=fixG(self.DiGraphObject.GetGraph())
        self.DiGraph_check='false'
        self.PrunedDiGraph_check='false'
        self.Tree_check='false'    

        self.__UpdateGraphPlot()


        
    def _Prune_directed_graph_fired(self):
        
        if self.Graph.is_directed():
            pass
        else:
            print('Directed graph is needed!')
            return
        
        try: 
            self.DiGraphObject
        except:return
        
        
        if len(self.Sources)==0 and len(self.Sinks)==0:
            print('Sources and/or Sinks need to be set!')
            
        else:
            
            end_nodes_to_exclude=[i for i in self.Sources]
            end_nodes_to_exclude.extend(self.Sinks)
            
            self.DiGraphObject.CloseEnds(end_nodes_to_exclude)
            self.PrunedDiGraph_check='true'
            self.Tree_check='false'
    
            self.Graph=fixG(self.DiGraphObject.GetDiGraph())
            self.__UpdateGraphPlot()


    def _Propagate_vessel_type_fired(self):
        
        if self.Graph.is_directed():
            pass
        else:
            print('Directed graph is needed!')
            return
        
        try: 
            self.DiGraphObject
        except:return

        done=0
        
        if self.Vessel_type=='Artery':
            try:
                self.DiGraphObject.PropagateTypes(Starting_nodes=self.Sources, 
                                              cutoff=self.Propagate_cutoff,
                                              value=self.TypeValue[self.Vessel_type])   
                done=1
            except: return
 
        if self.Vessel_type=='Vein':
            try:
                self.DiGraphObject.PropagateTypes(Starting_nodes=self.Sinks, 
                                              cutoff=self.Propagate_cutoff,
                                              value=self.TypeValue[self.Vessel_type], 
                                              backward=True) 
                done=1
            except: return
                                               
        if done:
            values_to_exlude=[self.TypeValue[i] for i in self.TypeValue.keys()]
            self.DiGraphObject.PropagateCapillaryTypes(value=self.TypeValue['Capillary'],
                                                       values_to_exlude=values_to_exlude)
            self.Graph=fixG(self.DiGraphObject.GetDiGraph())
            self.__UpdateGraphPlot()


    def _Convert_to_tree_graph_fired(self):
        
        if self.Graph.is_directed():
            pass
        else:
            print('Directed graph is needed!')
            return
        
        self.DiGraphObject.SetDiGraph(self.Graph)
        
        self.DiGraphObject.UpdateTreeFromDiGraph(root=self.Sources[0], forest=self.Forest)
        self.Tree_check='true'
        self.Graph=fixG(self.DiGraphObject.GetTree())
        self.__UpdateSourcesSinks()
        self.__UpdateGraphPlot()


    def _Label_branches_fired(self):
        
        if self.Graph.is_directed():
            pass
        else:
            print('Directed graph is needed!')
            return
        
        self.DiGraphObject.SetDiGraph(self.Graph)
        self.DiGraphObject.LabelDiGraphBranching2()
        self.Graph=fixG(self.DiGraphObject.GetDiGraph())
        self.__UpdateGraphPlot()
 
    def _Label_tree_branches_fired(self):
        
        if self.Graph.is_directed():
            pass
        else:
            print('Directed graph is needed!')
            return
        
        self.DiGraphObject.SetTree(self.Graph)
        self.DiGraphObject.LabelTreeBranching()
        self.Graph=fixG(self.DiGraphObject.GetTree())
        self.__UpdateGraphPlot()     
        
    @on_trait_change('Scene.activated')
    def initiate_scene(self):
        
        # initiate cam
        self.__SetCamDefault()

        #initiate pickers
        def node_picker_callback(picker):
            """ Picker callback: this get called when on pick events.
            """
            if picker.actor in self.GraphPlot.Glyph.actor.actors:
    
                self.NodeID = picker.point_id/self.GraphPlot.GetGlyphSourcePoints().shape[0]
                self.NodeID = str(int(float(self.NodeID))) # for python 3
                print('Node id: '+self.NodeID)
                self.UpdateNodeInfo()
                
                if self.NodeID != -1:
                    self.UpdateNodeOutline()
        
        self.NodePicker = self.Figure.on_mouse_pick(node_picker_callback)             
        self.NodePicker.tolerance = 0.01
        
# -------------------------------------------------------
# Functions
# -------------------------------------------------------
 
    # --- decorators ----#
    def __show_status(t):
        def decorator(f):
            def wrapper(i):
                t.replace(t,'')
                try:
                    t.join(str(f(i)))
                except:
                    t.join('Cannot update status!')
            return wrapper
        return decorator
    
    # -----------------------# 
    def UpdateNodeInfo(self):
        
        
        self.NodePos = str(np.round(self.Graph.node[int(self.NodeID)]['pos'], 3))
        
        
        NodeType=None
        NodeRadius=None
        NodeBranch=None
        NodeFlow=None
        
        try:
            NodeType = self.Graph.node[int(self.NodeID)]['type']
            self.NodeType=NodeType
        except: pass
    
        try:
            NodeRadius = np.round(self.Graph.node[int(self.NodeID)]['r'], 3)
            self.NodeRadius=NodeRadius
        except: pass

        try:
            NodeBranch= self.Graph.node[int(self.NodeID)]['branch']
            self.NodeBranch=NodeBranch
        except: pass
       
        try:    
            NodeFlow= self.Graph.node[int(self.NodeID)]['flow']
            self.NodeFlow=NodeFlow
        except: pass

        # construct status
        
        self.Status1='[Node: '+str(self.NodeID)+'] [Position: '+self.NodePos+'] '
        
        if NodeRadius is not None:
          self.Status1=self.Status1+'[Radius: '+str(self.NodeRadius)+'] '
        
        if NodeType is not None:
            self.Status1=self.Status1+'[Type: '+self.ValueType[int(float(self.NodeType))]+'] '
        
        if NodeFlow is not None:
            self.Status1=self.Status1+'[flow: '+str(self.NodeFlow)+'] '           
                    
   
    def __SetGraphAttr(self):
        
        # set type
        try: test=[self.Graph.node[i]['type'] for i in self.Graph.GetNodes()]
        except:
            for i in self.Graph.GetNodes():
                self.Graph.node[i]['type']=1    
        
        # set radius
        try: test=[self.Graph.node[i]['r'] for i in self.Graph.GetNodes()]
        except: 
            try:test=[self.Graph.node[i]['d'] for i in self.Graph.GetNodes()]
            except:
                for i in self.Graph.GetNodes():
                    self.Graph.node[i]['r']=1 
                    
        # set branch level
        try: test=[self.Graph.node[i]['branch'] for i in self.Graph.GetNodes()]
        except:
            for i in self.Graph.GetNodes():
                self.Graph.node[i]['branch']=1 
             
    def __UpdateGraph(self):
        
        self.__ResetGraphAll()
        
        check=self.GraphFilePath.split('.')
        
        if check[-1]=='swc':
            self.Graph=ReadSWC(self.GraphFilePath).GetOutput()
            
        if check[-1]=='pajek':
            if check[-2]=='di':
                self.Graph=ReadPajek(self.GraphFilePath, mode='di').GetOutput()
            else:
                self.Graph=ReadPajek(self.GraphFilePath).GetOutput()
        
        self.Graph=fixG(self.Graph)
        self.NNodes=self.Graph.number_of_nodes()
        self.__SetGraphAttr()
        self.DiGraphObject.SetGraph(self.Graph)
        
        
    def __UpdateSourcesSinks(self):
        
        self.Sources=[]
        self.Sinks=[]
        
        for i in self.Graph.GetNodes():
            try:
                if self.Graph.node[i]['source']=='1':
                    self.Sources.append(i)
            except: pass
                
        for i in self.Graph.GetNodes():
            try:
                if self.Graph.node[i]['sink']=='1':
                    self.Sinks.append(i) 
            except: pass


    def __SetDiGraph(self): pass
    def __SetTree(self): pass

    def __UpdateGraphPlot(self):
        
        if self.GraphPlot is None:
            self.GraphPlot=GraphPlot()
            self.NodeOutline=mlab.outline(line_width=3, color=(0,0,0))
            self.NodeOutline.bounds = (0, 0,
                                       0, 0,
                                       0, 0) 
            
        self.GraphPlot.Update(self.Graph)
        
    def __UpdateStack(self):
        
        self.Stack=ReadStackMat(self.StackFilePath).GetOutput()>0
        self.Stack=self.Stack.astype('int')
    
    def __UpdateStackPlot(self):
        
        if self.StackPlot is None:
            self.StackPlot=StackPlot()
        
        if np.size(self.Stack)==0: 
            print('No Stack found!')
        else:
            self.StackPlot.Update(self.Stack)

    def __RemoveGraphPlot(self):
        self.GraphPlot.Remove()
        self.GraphPlot=None
      
    def __RemoveStackPlot(self):
        self.StackPlot.Remove()
        self.StackPlot=None 
        
    def __ResetGraphAll(self):
        
        # calc
        self.Graph=None
        self.DiGraphObject=AnnotateDiGraph(self.Graph)
        self.DiGraph_check='false'
        self.PrunedDiGraph_check='false'
        self.Tree_check='false'
        self.Sources=[]
        self.Sinks=[]
        self.Status1=''

        # vis
        try:
            self.TubeRadiusByScale=False
            self.TubeRadiusByColor=False
            self.TubeTypeByColor=False
            self.TubeBranchingByColor=False
            self.TubeFlowByColor=False
            self.TubeVelocityByColor=False
            self.TubePressureByColor=False
            self.NodeOutline.remove()
            self.__RemoveGraphPlot()
        except: pass
        
    def UpdateNodeOutline(self):
        
        if self.Graph is not None:
            
            if self.NodeID:
                x, y, z = self.Graph.node[int(self.NodeID)]['pos'] 
            else:
                x, y, z= self.Graph.node[0]['pos'] 
                
            self.NodeOutline.bounds = (x-self.Selection_size, x+self.Selection_size,
                                       y-self.Selection_size, y+self.Selection_size,
                                       z-self.Selection_size, z+self.Selection_size)         
        
    def __UpdateSkeleton(self):
        
        try:
            self.__ResetGraphAll()
            self.__RemoveGraphPlot()
        except: pass
        
        t0=time()
        
        generate=GenerateGraph(self.Stack)
        
        if self.Initial_graph=='Grid':
            generate.UpdateGridGraph(Sampling=self.Sampling)
            GeneratedGraph=generate.GetOutput()
        else:
            generate.UpdateRandomGraph(connection=int(self.Connections), 
                                       nInitialNodes=int(self.InitNodes))
            GeneratedGraph=generate.GetOutput()
        
        
        contract=ContractGraph(GeneratedGraph)
        contract.Update(DistParam=self.DistParam, 
                        MedParam=self.MedParam, 
                        SpeedParam=self.SpeedParam, 
                        DegreeThreshold=float(self.Angle), 
                        ClusteringResolution=float(self.ClusteringRes),
                        StopParam=float(self.StopParam),
                        NFreeIteration=int(self.FreeIter))
        ContractedGraph=contract.GetOutput()        

        refine=RefineGraph(ContractedGraph)
        refine.Update(AreaParam=float(self.PolyArea), 
                      PolyParam=int(self.PolyNum))
        self.Graph=refine.GetOutput()        
        #self.Graph=prunG(self.Graph)
        
        print('Number of iterations: '+str(contract.Iteration))
        print('Time to generate the model: '+ str(time()-t0))
        
        
        self.__SetGraphAttr()
        self.DiGraphObject.SetGraph(self.Graph)
        self.__UpdateGraphPlot()
    
    def __UpdateRadius(self, fixed=False):
        
        refine=RefineGraphRadius(self.Graph)

        if fixed==False:
            
            if self.Graph.is_directed():
                refine.UpdateRefineRadiusDirected(Cutoff=int(self.Degree_cutoff), 
                                                  Mode=self.Smoothing_mode)
            else:
                refine.UpdateRefineRadius(Cutoff=int(self.Degree_cutoff), 
                                          Mode=self.Smoothing_mode)            
        else:
            if self.Graph.is_directed():
                refine.UpdateFixedRadiusOnBranches(Mode=self.Fixing_mode,
                                                  DictDirectedBranches=\
                                                  self.DiGraphObject.DictDirectedBranches)
            else:
                print('Works on direcetd graghs only!')
                
        self.Graph=refine.GetOutput()
        self.__UpdateGraphPlot()

        
    def __ResetSources(self):
        self.Sources=[]
        self.Sinks=[]
        
    def __SetCamDefault(self): 
        
        self.MyScene.camera.position = [-75, -1000, -250]
        self.MyScene.camera.focal_point = [250, 250, 300]
        self.MyScene.camera.view_angle = 30.0
        self.MyScene.camera.view_up = [0.25, 0.5, -1]
        self.MyScene.camera.clipping_range = [716.7128115564021, 2576.5923162091494]
        self.MyScene.camera.compute_view_plane_normal()


    def __SetCamParam(self, position, 
                      focal_point, 
                      view_angle, 
                      view_up, 
                      clipping_range): 
        
        self.MyScene.camera.position = position
        self.MyScene.camera.focal_point = focal_point
        self.MyScene.camera.view_angle = view_angle
        self.MyScene.camera.view_up = view_up
        self.MyScene.camera.clipping_range = clipping_range
        self.MyScene.camera.compute_view_plane_normal()
                
      
    def __SaveCamParam(self, path, name):
        
        position = self.MyScene.camera.position 
        focal_point = self.MyScene.camera.focal_point
        view_angle = self.MyScene.camera.view_angle 
        view_up = self.MyScene.camera.view_up 
        clipping_range = self.MyScene.camera.clipping_range

        position='position: '+str(position[0])+' '+str(position[1])+' '+str(position[2])
        focal_point='focal_point: '+str(focal_point[0])+' '+str(focal_point[1])+' '+str(focal_point[2])
        view_angle='view_angle: '+str(view_angle)
        view_up='view_up: '+str(view_up[0])+' '+str(view_up[1])+' '+str(view_up[2])
        clipping_range='clipping_range: '+str(clipping_range[0])+' '+str(clipping_range[1])

        f=open(path+name, 'w')
        lines=[position,'\n', 
               focal_point,'\n', 
               view_angle,'\n', 
               view_up, '\n',
               clipping_range]
        
        f.writelines(lines)
        f.close()


    def __LoadCamParam(self, path, name):
        
        f=open(path+name, 'r')
        lines=f.readlines()
        
        position = lines[0].split(' ')
        position=position[1:]
        position=[float(i) for i in position]
    
        focal_point = lines[1].split(' ')
        focal_point=focal_point[1:]
        focal_point=[float(i) for i in focal_point]
        
        view_angle =lines[2].split(' ')
        view_angle=float(view_angle[1])
        
        
        view_up = lines[3].split(' ')
        view_up=view_up[1:]
        view_up=[float(i) for i in view_up]
    
    
        clipping_range = lines[4].split(' ')
        clipping_range=clipping_range[1:]
        clipping_range=[float(i) for i in clipping_range]
        
        f.close()

        self.__SetCamParam(position, focal_point, 
                           view_angle, view_up, clipping_range)
        
     
        
    def __Run(self):
        self.configure_traits()
        
#------------------------------------------------------------
#       CONTROL GROUP
#------------------------------------------------------------
    control_group = Group(
            
                       #------------------------------------
                        Group(Group(Item('Surface_opacity'),
                           Item('Surface_color'), 
                           orientation='horizontal',
                           layout='normal'), label='Stack'),
                       #------------------------------------
                        Group(        
                           Group(Item('Gylph_size'), 
                                 Item('Gylph_color'),
                                 orientation='horizontal',
                                 layout='normal'),
                           Group(Item('Tube_radius'), 
                                 Item('Tube_color'), 
                                 orientation='horizontal',
                                 layout='normal'),
                           Group(
                                 Item('TubeRadiusByScale'), 
                                 Item('TubeRadiusByColor'),
                                 Item('TubeTypeByColor'),
                                 Item('TubeBranchingByColor'),
                                 orientation='horizontal',
                                 layout='normal'), 
                           Group(
                                 Item('TubeFlowByColor'),
                                 Item('TubeVelocityByColor'),
                                 Item('TubePressureByColor'),
                                 orientation='horizontal',
                                 layout='normal'),                                    
                            label='Graph'),
                       #------------------------------------
                        Group(Item('Selection_size'), 
                              orientation='horizontal',
                              layout='normal'), 
                        
                    label='Control', layout='normal')

#------------------------------------------------------------
#       Input/Ouptut GROUP
#------------------------------------------------------------
                        
    IO_group =      Group( 
                        Group(Item('Load_stack', show_label=False), 
                           Item('Remove_stack', show_label=False), 
                           orientation='horizontal'),
                       #------------------------------------
                        Group(Item('Load_graph', show_label=False), 
                           Item('Remove_graph', show_label=False), 
                           orientation='horizontal'),
                       #------------------------------------
                       Group(Item('Save_graph', show_label=False),
                             Item('MetaData'),
                             Item('Save_type', label='Save as'),
                             orientation='horizontal'),
                             
                        Group(Item('Load_camera', show_label=False), 
                           Item('Save_camera', show_label=False), 
                           orientation='horizontal'),                             
                         
                        Group(Item('Save_snapshot', show_label=False), 
                           Item('Snapshot_resolution', label='Resolution', show_label=True), 
                           orientation='horizontal'),                                
                              
                    label='I/O', layout='normal') 


#------------------------------------------------------------
#       Skel GROUP
#------------------------------------------------------------

    skel_group = Group(
            
                    Group(
                        Group(
                           Item('Sampling'), 
                           Item('SpeedParam'),
                           Item('DistParam'),
                           Item('MedParam'),
                           layout='normal', label='Main paramters', show_border = True),  
                        
                        Group(  
                            Group(Item('Initial_graph'),
                                  Item('InitNodes'),
                                  Item('Connections'), 
                            orientation='horizontal'),
                                  
                            Item('StopParam'),
                            Item('FreeIter'),
                            Item('ClusteringRes'),
                            Item('Angle'),
                            Item('PolyArea'),
                            Item('PolyNum'),
                            layout='normal', label='Extra paramters', show_border = True),  
                                
                        orientation='horizontal'),
                        
                    Item('SetDefault', show_label=False),
                    Item('Skeletonize', show_label=False),
                    layout='normal', label='Skeleton', orientation='vertical', show_border = True)




#------------------------------------------------------------
#       Edit GROUP
#------------------------------------------------------------
                     
    edit_group =  Group(
                     
                     
                         Group(Item('Smooth_radius', show_label=False),
                               Item('Degree_cutoff'),
                               Item('Smoothing_mode'),
                               orientation='horizontal', label='Radius', show_border=True,
                                ), 
                         
                        Group(Item('Fix_radius', show_label=False),
                               Item('Fixing_mode'),
                               orientation='horizontal'
                                ),                                
                         Group(
                             Group(
                                Item('Remove_node', show_label=False), 
                                
                                Group(Item('NodeID', label='Active node', editor=TitleEditor()),
                                      Item('NodeType', label='Type', editor=TitleEditor()),
                                      orientation='vertical'),                                
                                
                                Group(Item('NodePos', label='Position', editor=TitleEditor()),
                                      Item('NodeRadius', label='Radius', editor=TitleEditor()),
                                      orientation='vertical'),                                      
                                      
                                orientation='horizontal'),
                             
                             Group(
                                     Item('nodes_list', style='readonly', label='Selected nodes'),
                                     Item('reset_nodes', show_label=False),
                                     Item('connect_nodes', show_label=False),
                                     orientation='horizontal'
                                    ),
                            label='Toplogy', show_border = True),
                                                  
                     label='Edit', orientation='vertical', layout='normal',show_border = True)
 
#------------------------------------------------------------
#       Labeling GROUP
#------------------------------------------------------------
                     
    labeling_group =  Group(

                        Group(     
                                Group(Item('Add_source', show_label=False),
                                      Item('Remove_source', show_label=False),
                                      Item('Add_sink', show_label=False),
                                      Item('Remove_sink', show_label=False),
                                      Item('Labeling_reset', show_label=False),
                                      orientation='vertical'),
                                
                                Item('Sources'),
                                Item('Sinks'),
                                orientation='horizontal', show_border=True, label='Sources/Sinks',
                                ),

                        Group(
                                Group(
                                    Group(
                                        Item('Generate_directed_graph', show_label=False),
                                        Item('DiGraph_check', style='readonly', label='Status'),
                                        orientation='horizontal'),
                                    Group(
                                        Item('Prune_directed_graph', show_label=False),
                                        Item('PrunedDiGraph_check', style='readonly', label='Status'),
                                        orientation='horizontal'),
                                    orientation='horizontal'), 
                                            
                                Item('Label_branches', show_label=False), 
                                Item('Generate_and_label_directed_graph', show_label=False),                                   
                                Item('Recall_undirected_graph', show_label=False),
                                             
                                Group( 
                                    Item('Propagate_vessel_type', show_label=False),
                                    Item('Propagate_cutoff', label='Depth'),
                                    Item('Vessel_type'),
                                    orientation='horizontal', show_border=True),

                                orientation='vertical', show_border=True, label='Directed graph from labeling'),
                        Group(
                            Item('Convert_to_tree_graph', show_label=False),
                            Item('Tree_check', style='readonly', label='Status'),
                            Item('Forest', label='Forest'),
                            Item('Label_tree_branches', show_label=False),
                            orientation='horizontal', label='Tree'), 
                                
                        label='Labeling', orientation='vertical', layout='normal',show_border = True)
                         


#------------------------------------------------------------
#       Labeling GROUP
#------------------------------------------------------------
                     
    flow_group =  Group(


                        Group(
                                Group(
                                    Item('Compute_flow', show_label=False),
                                    Item('Flow_check', style='readonly', label='Status'),
                                    orientation='horizontal'),
                                         
                                Group(
                                    Item('Compute_resistances', show_label=False),
                                    Item('Resistances_check', style='readonly', label='Status'),
                                    orientation='horizontal'),                                        

                                orientation='vertical', show_border=True, label='Flow'),

                        Group(
                                Group(
                                    Item('Generate_directed_graph_f', show_label=False),
                                    Item('DiGraph_check_f', style='readonly', label='Status'),
                                    orientation='horizontal'),
                                            
                                Item('Label_branches_f', show_label=False),                                    
                                Item('Recall_undirected_graph_f', show_label=False),
                                             
                                Group( 
                                    Item('Propagate_vessel_type_f', show_label=False),
                                    Item('Propagate_cutoff_f', label='Depth'),
                                    Item('Vessel_type_f'),
                                    orientation='horizontal', show_border=True),

                                orientation='vertical', show_border=True, label='Directed graph from flow'),
                                
                        label='Blood flow', orientation='vertical', layout='normal',show_border = True)
                         



#------------------------------------------------------------
#    #########    V I E W    #########
#------------------------------------------------------------

    traits_view = View(
                    Group(
        
                           Group(
                           
                                Group(
                                    control_group, IO_group, 
                                    label='Control', orientation='vertical', layout='split'),
                                
                                skel_group,
                                edit_group,
                                labeling_group,
                                flow_group,
                                layout='tabbed'
                                ),
        
                            Group(
                                Item('Status1', label='', show_label=False, editor=TitleEditor()),
                                Item('Scene', editor=SceneEditor(scene_class=MayaviScene),
                                       height=480, width=720, show_label=False)),
                                
                        layout='split', orientation='horizontal'),
                
                resizable=True,    
                title = 'GraphLab'    
               )   

        
if __name__=='__main__':
       
    window=MainDialogue()

#    path='/home/rdamseh/GraphPaper2018V1/rafat.swc'   
#    g=ReadSWC(path)
#    g=g.GetOutput()
    
#    import VascGraph.Tools.VisTools as v
#    from mayavi import mlab
#    mlab.figure()
#    v.visG(fixG(g))
#    
#    pathw='/home/rdamseh/GraphPaper2018V1/'
#    name='rafat'
#    
#    w=WriteSWC(pathw, name, g)
#        
    
    
    
#     g_p=window.GraphPlot.GetGlyphSourcePoints()
#     t_g=window.GraphPlot.GetTubePoints()




#     s=window.GraphPlot.Tube.get_output_dataset().point_data.scalars.to_array()
#     import numpy as np
#     s_=np.linspace(1, 10.0, len(s))
#     window.GraphPlot.Tube.get_output_dataset().point_data.scalars=s_
#     window.GraphPlot.Tube.update_data()
    