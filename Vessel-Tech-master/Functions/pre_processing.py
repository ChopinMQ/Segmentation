import numpy as np
from PIL import Image
import cv2
from Functions.BM3D import BM3D_1st_step
import bm3d
from Functions.help_functions import *
import numpy as np
from skimage import data
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import math
from typing import Optional
from scipy.fftpack import fft2, ifft2

from Functions.psf2otf import psf2otf



#My pre processing (use for both training and testing!)
def my_PreProc(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data) 
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    #temp = bpdfhe(train_imgs[0, 0, :, :])
    #train_imgs[0, 0, :, :] = temp
    #train_imgs = clahe_equalized(train_imgs)
    #train_imgs = adjust_gamma(train_imgs, 1.2)
    #blurred = np.float32(train_imgs)
    #blurred = anisodiff(blurred[0,0,:,:], niter=500, kappa=20, option=1, gamma=0.25, step=(1, 1))
    #train_imgs[0,0,:,:] = blurred
    blurred = L0Smoothing(train_imgs[0,0,:,:], param_lambda=0.04).run()
    train_imgs[0,0,:,:] = np.squeeze(blurred)/np.max(np.squeeze(blurred))*255
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = denoise(train_imgs)
    
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs

def my_PreProc_filo(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    print(data.shape)
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    #temp = bpdfhe(train_imgs[0, 0, :, :])
    #train_imgs[0, 0, :, :] = temp
    #train_imgs = clahe_equalized(train_imgs)
    train_imgs = train_imgs.astype("uint8")
    #blurred = cv2.bilateralFilter(train_imgs[0,0,:,:], -1, 75, 9)
    blurred = L0Smoothing(train_imgs[0,0,:,:], param_lambda=0.02).run()
    train_imgs[0,0,:,:] = np.squeeze(blurred)/np.max(np.squeeze(blurred))*255
    #temp = bpdfhe(train_imgs[0, 0, :, :])
    #train_imgs[0, 0, :, :] = temp
    #train_imgs = clahe_equalized(train_imgs)

    train_imgs = adjust_gamma(train_imgs, 1.2)
    #train_imgs = denoise(train_imgs)
    
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    
    return train_imgs

#==== histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], 
        dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        if np.max(imgs_normalized[i])-np.min(imgs_normalized[i])>0:
            imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) /
             (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
        else:
            imgs_normalized[i] = (imgs_normalized[i]-np.min(imgs_normalized[i]))*255
    return imgs_normalized



def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

def denoise(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    
    new_imgs = np.empty(imgs.shape)
    print("$%$$%$%$%$%$%$%$%$%$")
    print(imgs.shape)
    for i in range(imgs.shape[0]):

        #Basic_img = BM3D_1st_step(imgs[i,0])
        #new_imgs[i] = Basic_img
        #new_imgs[i] = imgs[i,0]
        new_imgs[i] = bm3d.bm3d(imgs[i,0], sigma_psd=2.7*25,stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return new_imgs


def bpdfhe(image):
    
    opimg = np.zeros_like(image)
 
    array1 = np.zeros(256*4+1, dtype = int)
    hist = array1[1:257]
    deltahist = array1[257:513]
    delta2hist = array1[513:769]
    histmax = array1[769:]
    

    hist, bin_edges = np.histogram(image, bins=256)

    x_qual = np.arange(0, 11, 1)

    membership= fuzz.trimf(x_qual, [0, 5, 10])

    fuzzyhist = np.zeros((np.size(hist)+np.size(membership)-1,1)).T;


    for counter in range(np.size(membership)):

        fuzzyhist = fuzzyhist + membership[counter]*np.concatenate((np.zeros(counter+1-1), hist, np.zeros(np.size(membership)-counter-1)),axis=0)

    fuzzyhist = fuzzyhist.T[math.ceil(np.size(membership)/2):-math.floor(np.size(membership)/2)+1]

    for i in range(255):
        deltahist[i] = (fuzzyhist[i+1]-fuzzyhist[i-1])/2
        delta2hist[i] = fuzzyhist[i+1]-2*fuzzyhist[i]+fuzzyhist[i-1]
    for i in range(255):
        if (deltahist[i+1]*deltahist[i-1]<0 and delta2hist[i]<0):
            histmax[i] = 1

    parts = np.where(histmax)[0]
    x = np.append(0,parts)
    x = np.append(x,255)

    span = np.zeros(x.shape[0], dtype = float) 
    M = np.zeros(x.shape[0], dtype = float) 
    factor = np.zeros(x.shape[0], dtype = float) 
    rang = np.zeros(x.shape[0], dtype = float) 
    start = np.zeros(x.shape[0], dtype = float) 

    Msum = np.cumsum(hist)

    for i in range(x.shape[0]):
        span[i-1]   = x[i] - x[i-1]
        if (span[i-1]<0):
            span[i-1]=1
        M[i-1] = Msum[x[i]]-Msum[x[i-1]]
        if (M[i-1]<=0):
            M[i-1]=1
        factor[i-1] = span[i-1]*math.log10(abs(M[i-1]))

    factorsum = sum(factor)

    for i in range(x.shape[0]):    
        rang[i-1] = 255*factor[i-1]/factorsum

    start = np.cumsum(rang)
    start = np.append(0,start)
    start = start.astype(int)

    opimg = np.zeros_like(image)
    img2 = np.zeros_like(image)
    small = np.amin(image)
    big = np.amax(image)

    y = np.zeros(hist.shape, dtype = float) 
    for i in range(start.shape[0]):

        y[start[i-1]:start[i]] = start[i-1]+rang[i-1]/M[i-1]*np.cumsum(hist[start[i-1]:start[i]])

    opimg = y[image.astype(int)]
    
    return opimg


def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),option=1,ploton=False):
    """
    Anisotropic diffusion.
 
    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)
 
    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration
 
    Returns:
            imgout   - diffused image.
 
    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.
 
    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)
 
    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes
 
    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.
 
    Reference: 
    P. Perona and J. Malik. 
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    12(7):629-639, July 1990.
 
    Original MATLAB code by Peter Kovesi  
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>
 
    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>
 
    June 2000  original version.       
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """
 
    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)
 
    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()
 
    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()
 
    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep
 
        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
 
        ax1.imshow(img,interpolation='nearest')
        ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")
 
        fig.canvas.draw()
 
    for ii in range(niter):
 
        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)
 
        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
 
        # update matrices
        E = gE*deltaE
        S = gS*deltaS
 
        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]
 
        # update the image
        imgout += gamma*(NS+EW)
 
        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)
 
    return imgout
 
def anisodiff3(stack,niter=1,kappa=50,gamma=0.1,step=(1.,1.,1.),option=1,ploton=False):
    """
    3D Anisotropic diffusion.
 
    Usage:
    stackout = anisodiff(stack, niter, kappa, gamma, option)
 
    Arguments:
            stack  - input stack
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (z,y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the middle z-plane will be plotted on every 
                 iteration
 
    Returns:
            stackout   - diffused stack.
 
    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.
 
    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)
 
    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x,y and/or z axes
 
    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.
 
    Reference: 
    P. Perona and J. Malik. 
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    12(7):629-639, July 1990.
 
    Original MATLAB code by Peter Kovesi  
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>
 
    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>
 
    June 2000  original version.       
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """
 
    # ...you could always diffuse each color channel independently if you
    # really want
    if stack.ndim == 4:
        warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
        stack = stack.mean(3)
 
    # initialize output array
    stack = stack.astype('float32')
    stackout = stack.copy()
 
    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()
 
    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep
 
        showplane = stack.shape[0]//2
 
        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
 
        ax1.imshow(stack[showplane,...].squeeze(),interpolation='nearest')
        ih = ax2.imshow(stackout[showplane,...].squeeze(),interpolation='nearest',animated=True)
        ax1.set_title("Original stack (Z = %i)" %showplane)
        ax2.set_title("Iteration 0")
 
        fig.canvas.draw()
 
    for ii in xrange(niter):
 
        # calculate the diffs
        deltaD[:-1,: ,:  ] = np.diff(stackout,axis=0)
        deltaS[:  ,:-1,: ] = np.diff(stackout,axis=1)
        deltaE[:  ,: ,:-1] = np.diff(stackout,axis=2)
 
        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gD = np.exp(-(deltaD/kappa)**2.)/step[0]
            gS = np.exp(-(deltaS/kappa)**2.)/step[1]
            gE = np.exp(-(deltaE/kappa)**2.)/step[2]
        elif option == 2:
            gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
            gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[2]
 
        # update matrices
        D = gD*deltaD
        E = gE*deltaE
        S = gS*deltaS
 
        # subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. don't as questions. just do it. trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:,: ,: ] -= D[:-1,:  ,:  ]
        NS[: ,1:,: ] -= S[:  ,:-1,:  ]
        EW[: ,: ,1:] -= E[:  ,:  ,:-1]
 
        # update the image
        stackout += gamma*(UD+NS+EW)
 
        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(stackout[showplane,...].squeeze())
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)
 
    return stackout
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