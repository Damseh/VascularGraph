#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:34:13 2019

@author: rdamseh
"""

import VascGraph.GraphIO as io


path='YuankangOCT/S5/Before/S5BeforeD_labeld.tif_graph.pajek'
root=28114

g=io.ReadPajek(path).GetOutput()
