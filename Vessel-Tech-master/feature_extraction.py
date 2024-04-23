#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wangxuelin
"""


import os
import math
import numpy, scipy.io
from PIL import Image
import matplotlib.colors as mcolors
import cv2
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import operator
from scipy.spatial.distance import euclidean
import pandas as pd
from Connectivity_matrix_test import *
from _utility import *
from skimage.morphology import skeletonize, medial_axis
from skimage.morphology import thin
import matplotlib.pyplot as plt
from skimage import morphology
import cv2
from typing import Optional
import math
from scipy.fftpack import fft2, ifft2

# perform morphological thinning of a binary image

def FE():

    imgs_dir='t/'
    save_dir='t/'
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):

            print ("original image: " +files[i])
            image = Image.open(imgs_dir+files[i])
            image = np.asarray(image)
            #image = cv2.bilateralFilter(image, -1, 75, 9)
            image = image.astype("uint8")
            ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_img = Image.fromarray(binary)
            binary_img.save(save_dir+"binary.jpg")
            binary[binary == 255] = 1
            skeleton0 = morphology.skeletonize(binary)
            skeleton = skeleton0.astype(np.uint8)
            image = Image.fromarray(image)

            graph=extract_graph(skeleton,image)
            print(graph.number_of_nodes())
            
            ############ merge nodes
            k=True

            while k:
                tmp=graph.number_of_nodes()
                attribute="line"
                # attribute_threshold_value=10
                attribute_threshold_value=5
                to_be_removed = [(u, v) for u, v, data in
                                            graph.edges(data=True)
                                            if operator.le(data[attribute],
                                                        attribute_threshold_value)]
                length=len(to_be_removed)
                for n in range(length):  
                    nodes=to_be_removed[n]
                    merge_nodes_2(graph,nodes)
                
                for n1,n2,data in graph.edges(data=True):
                    line=euclidean(n1,n2)
                    graph[n1][n2]['line']=line
                
                number_of_nodes=graph.number_of_nodes()
                k= tmp!=number_of_nodes
            
            print(graph.number_of_nodes())
            
            #Check connected
            ####  keep the connected with 1 poitns
            compnt_size = 0
            compnt_size = 1
            operators ="smaller or equal"
            oper_str_value = operators
            operators = operator.le
            connected_components = sorted(
                            # new version: (G.subgraph(c) for c in connected_components(G))
                            list(graph.subgraph(c) for c in nx.connected_components(graph)),
                            key=lambda graph: graph.number_of_nodes())
            
            to_be_removed = [subgraph for subgraph in connected_components
                                        if operators(subgraph.number_of_nodes(),
                                                                compnt_size)]
            for subgraph in to_be_removed:
                graph.remove_nodes_from(subgraph)
                        
            print ('discarding a total of', len(to_be_removed),
                            'connected components ...')
            
            ########
            
            nodes=[n for n in graph.nodes()]
            x=[x for (x,y) in nodes]
            y=[y for (x,y) in nodes]
            x1=int(np.min(x)+(np.max(x)-np.min(x))/2)
            y1=int(np.min(y)+(np.max(y)-np.min(y))/2)
            
            for n1,n2,data in graph.edges(data=True):
                centerdis1=euclidean((x1,y1),n2)
                centerdis2=euclidean((x1,y1),n1)
                #theta1=(math.atan2(-13,-14)/math.pi*180)%360
                #theta2=(math.atan2(-13,-14)/math.pi*180)%360
                
                if centerdis1>=centerdis2:
                    centerdislow=centerdis2
                    centerdishigh=centerdis1
                else:
                    centerdislow=centerdis1
                    centerdishigh=centerdis2
                graph[n1][n2]['centerdislow']=centerdislow
                graph[n1][n2]['centerdishigh']=centerdishigh
            
            
            
            ##############
            alldata=save_data(graph,center=False)
            data_name = files[i][0:6] + "_alldata.xlsx"
            writer = save_dir+data_name
            #writer = pd.ExcelWriter('/Users/wangxuelin/Downloads/STARE-im/im0324_alldata.xlsx', engine='xlsxwriter')

            alldata.to_excel(writer,index=False)
            
            degreedata=save_degree(graph,x1,y1)
            degree_name = files[i][0:6] + "_degreedata.xlsx"
            print ("degree data name: " + degree_name)

            #writer = pd.ExcelWriter('/Users/wangxuelin/Downloads/STARE-im/im0324_alldata.xlsx', engine='xlsxwriter')
            degreewriter = pd.ExcelWriter(save_dir+degree_name, engine='xlsxwriter')

            degreedata.to_excel(degreewriter,index=False)
            degreewriter._save()
            NODESIZESCALING = 30
            EDGETRANSPARENCYDIVIDER = 5
            pic=draw_graph2(np.asarray(image.convert("RGB")), graph,center=False)
            pic_name = files[i][0:6] + "_network.png"
            print ("pic name: " + save_dir + pic_name)
            plt.imshow(pic)
            plt.imsave(save_dir + pic_name, pic)
            
            

    return 0



class L0Smoothing:

    """Docstring for L0Smoothing. """

    def __init__(self, imgss,
                 param_lambda: Optional[float] = 2e-2,
                 param_kappa: Optional[float] = 2.0):
        """Initialization of parameters """
        self._lambda = param_lambda
        self._kappa = param_kappa
        self._imgss = imgss
        self._beta_max = 1e5

    def run(self, isGray=False):
        """L0 smoothing imlementation"""
        img = self._imgss
        S = img / 256
        if S.ndim < 3:
            S = S[..., np.newaxis]

        N, M, D = S.shape

        beta = 2 * self._lambda

        psf = np.asarray([[-1, 1]])
        out_size = (N, M)
        otfx = psf2otf(psf, out_size)
        psf = np.asarray([[-1], [1]])
        otfy = psf2otf(psf, out_size)

        Normin1 = fft2(np.squeeze(S), axes=(0, 1))
        Denormin2 = np.square(abs(otfx)) + np.square(abs(otfy))
        if D > 1:
            Denormin2 = Denormin2[..., np.newaxis]
            Denormin2 = np.repeat(Denormin2, 3, axis=2)

        while beta < self._beta_max:
            Denormin = 1 + beta * Denormin2

            h = np.diff(S, axis=1)
            last_col = S[:, 0, :] - S[:, -1, :]
            last_col = last_col[:, np.newaxis, :]
            h = np.hstack([h, last_col])

            v = np.diff(S, axis=0)
            last_row = S[0, ...] - S[-1, ...]
            last_row = last_row[np.newaxis, ...]
            v = np.vstack([v, last_row])

            grad = np.square(h) + np.square(v)
            if D > 1:
                grad = np.sum(grad, axis=2)
                idx = grad < (self._lambda / beta)
                idx = idx[..., np.newaxis]
                idx = np.repeat(idx, 3, axis=2)
            else:
                grad = np.sum(grad, axis=2)
                idx = grad < (self._lambda / beta)

            h[idx] = 0
            v[idx] = 0

            h_diff = -np.diff(h, axis=1)
            first_col = h[:, -1, :] - h[:, 0, :]
            first_col = first_col[:, np.newaxis, :]
            h_diff = np.hstack([first_col, h_diff])

            v_diff = -np.diff(v, axis=0)
            first_row = v[-1, ...] - v[0, ...]
            first_row = first_row[np.newaxis, ...]
            v_diff = np.vstack([first_row, v_diff])

            Normin2 = h_diff + v_diff
            # Normin2 = beta * np.fft.fft2(Normin2, axes=(0, 1))
            Normin2 = beta * fft2(Normin2, axes=(0, 1))

            FS = np.divide(np.squeeze(Normin1) + np.squeeze(Normin2),
                           Denormin)
            # S = np.real(np.fft.ifft2(FS, axes=(0, 1)))
            S = np.real(ifft2(FS, axes=(0, 1)))
            if False:
                S_new = S * 256
                S_new = S_new.astype(np.uint8)
                cv2.imshow('L0-Smooth', S_new)
                cv2.waitKey(0)

            if S.ndim < 3:
                S = S[..., np.newaxis]
            beta = beta * self._kappa

        return S
