import cv2
import numpy
import operator
import numpy, scipy.io


def draw_graph(image, graph):

    tmp = draw_edges(image, graph)
    
    return draw_nodes(tmp, graph, node_size)


def draw_nodes(img, graph, radius=1, center=True):
    
    if center:
        cv2.rectangle(img, (y1 - radius, x1 - radius), (y1 + radius, x1 + radius),
                         (  255,   0, 0), -1)
     
    for x, y in graph.nodes():
        if graph.degree[(x, y)]==1:
            cv2.rectangle(img, (y - radius, x - radius), (y + radius, x + radius),
                     (  0,   0, 255), -1)
        elif graph.degree[(x, y)]==2:
            cv2.rectangle(img, (y - radius, x - radius), (y + radius, x + radius),
                     (  0,  85, 170), -1)
        elif graph.degree[(x, y)]==3:
            cv2.rectangle(img, (y - radius, x - radius), (y + radius, x + radius),
                     (  0, 170,  85), -1)
        elif graph.degree[(x, y)]==4:
            cv2.rectangle(img, (y - radius, x - radius), (y + radius, x + radius),
                    (  0, 255,   0), -1)
        elif graph.degree[(x, y)]==5:
            cv2.rectangle(img, (y - radius, x - radius), (y + radius, x + radius),
                     ( 84, 170,   0), -1)
        elif graph.degree[(x, y)]==6:
            cv2.rectangle(img, (y - radius, x - radius), (y + radius, x + radius),
                     (170,  84,   0), -1)
        elif graph.degree[(x, y)]==7:
            cv2.rectangle(img, (y - radius, x - radius), (y + radius, x + radius),
                     (255,   0,   0), -1)
    
    return img


def draw_edges(img, graph, col=(255, 255, 255)):
    
    edg_img = numpy.copy(img)

    max_standard_deviation = 0

    for (x1, y1), (x2, y2) in graph.edges():
        start = (y1, x1)
        end = (y2, x2)
        diam = 1
        width_var = 1
        if diam == -1: diam = 2
        diam = int(round(diam))
        if diam > 255:
            print('Warning: edge diameter too large for display. Diameter has been reset.')
            diam = 255
        cv2.line(edg_img, start, end, col, diam)

    edg_img = cv2.addWeighted(img, 0.3, edg_img, 0.7, 0)
    return edg_img

def check_operator(dropdown):

    op_object = None

    if dropdown.value == "strictly smaller":
        op_object = operator.lt
    if dropdown.value == "smaller or equal":
        op_object = operator.le
    if dropdown.value == "equal":
        op_object = operator.eq
    if dropdown.value == "greater or equal":
        op_object = operator.ge
    if dropdown.value == "strictly greater":
        op_object = operator.gt
    return op_object

def draw_graph2(image, graph, center=True):

    tmp1 = draw_edges(image, graph)
    tmp= draw_curve(tmp1, graph)
    #tmp=image
    node_size=3
    return draw_nodes(tmp, graph, node_size,center)

def draw_curve(image, graph):
    
    for (x1, y1), (x2, y2),curve in graph.edges(data='curve'):
        for (u,v) in curve:
            cv2.rectangle(image, (v, u), (v, u),(255,0,0), -1)
    
    return image



