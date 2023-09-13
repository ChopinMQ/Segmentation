#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wangxuelin
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from PIL import Image
import cv2
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from collections import defaultdict
from itertools import chain
import pandas as pd


def zhang_suen_node_detection(skel):

    def check_pixel_neighborhood(x, y, skel):

        accept_pixel_as_node = False
        item = skel.item
        p2 = item(x - 1, y) / 255
        p3 = item(x - 1, y + 1) / 255
        p4 = item(x, y + 1) / 255
        p5 = item(x + 1, y + 1) / 255
        p6 = item(x + 1, y) / 255
        p7 = item(x + 1, y - 1) / 255
        p8 = item(x, y - 1) / 255
        p9 = item(x - 1, y - 1) / 255

        components = (p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) + \
                     (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) + \
                     (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) + \
                     (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1)

        #components = (p2 == 1) + (p4 == 1) + (p6 == 1) + (p8 == 1)
        if (components >= 3) or (components == 1):
            accept_pixel_as_node = True
        return accept_pixel_as_node

    graph = nx.Graph()
    w, h = skel.shape
    item = skel.item
    for x in range(1, w - 1):
        for y in range(1, h - 1):
             if item(x, y) != 0 and check_pixel_neighborhood(x, y, skel):
                graph.add_node((x, y))
    return graph


def breadth_first_edge_detection2(skel, segmented, graph):

    def neighbors(x, y):
        item = skel.item
        width, height = skel.shape
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if (dx != 0 or dy != 0) and \
                                        0 <= x + dx < width and \
                                        0 <= y + dy < height and \
                                item(x + dx, y + dy) != 0:
                    yield x + dx, y + dy

    def distance_transform_diameter(edge_trace, segmented):

        dt = cv2.distanceTransform(segmented, 2, 0)
        edge_pixels = np.nonzero(edge_trace)
        diameters = defaultdict(list)
        for label, diam in zip(edge_trace[edge_pixels], 2.0 * dt[edge_pixels]):
            diameters[label].append(diam)
        return diameters

    label_node = dict()
    label_pixel=dict()
    queues = []
    label = 1
    label_length = defaultdict(int)
    for x, y in graph.nodes():
        for a, b in neighbors(x, y):
            label_node[label] = (x, y)
            label_length[label] = 1.414214 if abs(x - a) == 1 and \
                                              abs(y - b) == 1 else 1
            label_pixel[label]=[(a,b)]
            queues.append((label, (x, y), [(a, b)]))
            label += 1


    edges = set()
    edge_trace = np.zeros(skel.shape, np.uint32)
    edge_value = edge_trace.item
    edge_set_value = edge_trace.itemset
    label_histogram = defaultdict(int)

    while queues:
        new_queues = []
        for label, (px, py), nbs in queues:
            for (ix, iy) in nbs:
                value = edge_value(ix, iy)
                if value == 0:
                    edge_set_value((ix, iy), label)
                    label_histogram[label] += 1
                    label_length[label] += 1.414214 if abs(ix - px) == 1 and \
                                                       abs(iy - py) == 1 else 1
                    label_pixel[label]=label_pixel[label]+[(a,b) for a,b in neighbors(ix, iy)]
                    new_queues.append((label, (ix, iy), neighbors(ix, iy)))
                elif value != label:
                    edges.add((min(label, value), max(label, value)))
        queues = new_queues

    diameters = distance_transform_diameter(edge_trace, segmented)
    for l1, l2 in edges:
        u, v = label_node[l1], label_node[l2]
        if u == v:
            continue
        d1, d2 = diameters[l1], diameters[l2]
        diam = np.fromiter(chain(d1, d2), np.uint, len(d1) + len(d2))
        graph.add_edge(u, v, pixels=label_histogram[l1] + label_histogram[l2],
                       length=label_length[l1] + label_length[l2],
                       curve=label_pixel[l1] + label_pixel[l2][::-1],
                       width=np.median(diam),
                       width_var=np.var(diam))
    return graph


def merge_nodes_2(G,nodes, attr_dict=None):
   
    if (nodes[0] in G.nodes()) & (nodes[1] in G.nodes()):
        for n1,n2,data in list(G.edges(data=True)):
            if (n1==nodes[0]) & (n2 != nodes[1]): 
                G.add_edges_from([(nodes[1],n2,data)])
            elif (n1!=nodes[1]) & (n2 == nodes[0]):
                G.add_edges_from([(n1,nodes[1],data)])
        G.remove_node(nodes[0])

def extract_graph(skeleton,image):
    
    skeleton=np.asarray(skeleton)
    skeleton=skeleton*255
    image=np.asarray(image)
    image=image.astype(np.uint8)
    image=image*255
    graph_nodes = zhang_suen_node_detection(skeleton)
    graph = breadth_first_edge_detection2(skeleton, image, graph_nodes)
    edges=[(u, v) for u, v, data in graph.edges(data=True)]
    data=[data for u, v, data in graph.edges(data=True)]

    for n1,n2,data in graph.edges(data=True):
        line=euclidean(n1,n2)
        graph[n1][n2]['line']=line

    return graph

def save_data(graph,center=False):
    lines=[]
    for n1,n2,data in graph.edges(data=True):
        lines.append(graph[n1][n2]['line'])
    
    length=[]
    for n1,n2,data in graph.edges(data=True):
        length.append(graph[n1][n2]['length'])
    
    width=[]
    for n1,n2,data in graph.edges(data=True):
        width.append(graph[n1][n2]['width'])
    
    width_var=[]
    for n1,n2,data in graph.edges(data=True):
        width_var.append(graph[n1][n2]['width_var'])
    
    nodespair=[(u, v) for u, v, data in graph.edges(data=True)]
    node1=[u for u, v, data in graph.edges(data=True)]
    node2=[v for u, v, data in graph.edges(data=True)]
    
    tortuosity=[x/y for x, y in zip(length,lines)]
    
    curve=[]
    for n1,n2,data in graph.edges(data=True):
        curve.append(graph[n1][n2]['curve'])

    
    if center:
        centerdislow=[]
        for n1,n2,data in graph.edges(data=True):
            centerdislow.append(graph[n1][n2]['centerdislow'])
            
        centerdishigh=[]
        for n1,n2,data in graph.edges(data=True):
            centerdishigh.append(graph[n1][n2]['centerdishigh'])
            
        thetalow=[]
        for n1,n2,data in graph.edges(data=True):
            thetalow.append(graph[n1][n2]['thetalow'])
            
        thetahigh=[]
        for n1,n2,data in graph.edges(data=True):
            thetahigh.append(graph[n1][n2]['thetahigh'])
            
        
    
        alldf=pd.DataFrame({'nodespair':nodespair,
                            'node1':node1,
                            'node2':node2,
                            'line':lines,
                            'length':length,
                            'width':width,
                            'width_var':width_var,
                            'tortuosity':tortuosity,
                            'centerdislow':centerdislow,
                            'centerdishigh':centerdishigh,
                            'thetalow':thetalow,
                            'thetahigh':thetahigh,
                            'curve':curve,})
    else:
        alldf=pd.DataFrame({'nodespair':nodespair,
                            'node1':node1,
                            'node2':node2,
                            'line':lines,
                            'length':length,
                            'width':width,
                            'width_var':width_var,
                            'tortuosity':tortuosity,
                            'curve':curve,})
    
    return alldf

def save_degree(graph,x,y):    
    degree=[]
    for n1,n2 in graph.nodes():
        degree.append(graph.degree[(n1, n2)])
    
    distance=[]
    for n1,n2 in graph.nodes():
        distance.append(euclidean((n1, n2),(x,y)))

    nodes=[]
    for n1,n2 in graph.nodes():
        nodes.append((n1, n2))
    
    nodedf=pd.DataFrame({'nodes':nodes,
                         'distance':distance,
                            'degree':degree,})
    return nodedf