import cv2
from PIL import Image
import numpy as np
import pandas as pd
import math
from typing import Optional
from scipy.fftpack import fft2, ifft2
import networkx as nx
import matplotlib.pyplot as plt
from ast import literal_eval as make_tuple

def get_clear_network(grey_path, xlsx_path, save_directory):
    img = Image.open(grey_path)
    img = np.asarray(img)
    image = img.astype("uint8")
    ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    data = pd.read_excel(xlsx_path)

    G = nx.Graph()

    for coor in data.nodespair:
        temp_c = make_tuple(coor)
        first = temp_c[0]
        second = temp_c[1]
        f = (first[1], first[0])
        s = (second[1], second[0])
        G.add_edge(f, s)

    x = [ val[0] for val in list(G.nodes())]
    y = [ val[1] for val in list(G.nodes())]

    components = list(nx.connected_components(G)) # get a list of connected small graphs

    # sort them and only keep the largest in the first position
    components.sort(key=len, reverse=True)

    largest = components.pop(0)

    contours, _ = cv2.findContours(image=binary, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

    def inner_loop(pt, contours):
        cts = []
        flag = 0
        seq = 0
        for ct in contours:
            seq += 1
            result = cv2.pointPolygonTest(ct, pt, False)
            if result >= 0:
            # it's inside the contour or on the boundary
                cts.append(ct)
                flag = 1
            if seq == len(contours) and flag == 1:
                return [1, cts]
            elif seq == len(contours) and flag == 0:
                return [0]

    binary_copy = binary.copy()
    count = 0
    for node_list in components:
        count += 1
        ele = list(node_list)
        try:
            for pt in ele:
                res = inner_loop(pt, contours)
                if res[0] == 1:
                    # it has a contour
                    # find the smallest
                    ar = []
                    tempc = res[1]
                    for ct in tempc:
                        area = cv2.contourArea(ct)
                        ar.append(area)
                    # get the index of the minimum
                    ind = np.argmin(ar)
                    cont_need = tempc[ind]
                    # fill this contour with black pixels
                    cv2.drawContours(binary_copy, [cont_need], -1, (0, 0, 0), thickness=cv2.FILLED)
        except:
            print(count-1)

    final_cont, _ = cv2.findContours(image=binary_copy, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    maxc = max(final_cont, key=cv2.contourArea)

    binary_f = binary_copy.copy()
    for ct in final_cont:
        if len(ct) != len(maxc):
            cv2.drawContours(binary_f, [ct], -1, (0, 0, 0), thickness=cv2.FILLED)

    final_cont, _ = cv2.findContours(image=binary_f, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    binary_final = binary_f.copy()
    for ct in final_cont:
        ar = cv2.contourArea(ct)
        if ar < 200:
            img_m = np.ones(binary_final.shape)*255
            img_m = img_m.astype("uint8")
            masks = cv2.drawContours(img_m, [ct], -1, (0, 0, 0), thickness=cv2.FILLED)
            binary_final = cv2.bitwise_and(masks, binary_final, mask=masks)
    save_directory = save_directory + '/tol_predictions.jpg'

    # save the image to a folder
    cv2.imwrite(save_directory, binary_final)