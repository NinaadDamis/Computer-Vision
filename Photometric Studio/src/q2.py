# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 12, 2022
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals, estimateShape, plotSurface 
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface 

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals
    
    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """
    u,s,v = np.linalg.svd(I,full_matrices = False)
    print("Shapes u,s,v ", u.shape,s.shape,v.shape)
    s[3:] = 0 # Zero out everything exxcept top K
    s = np.diag(s) # cpnvert to matrix
    I = u @ s @ v # Recompute
    u,s,v = np.linalg.svd(I,full_matrices = False)
    
    # We can combine s with either u or v. Doing with u.
    # compute B,L --> B shape = 3,159039 , L shape = 7,3
    B = v[:3,:]
    s = np.diag(s) # matrix
    s = s[:,:3] # to get L shape = 7,3
    L = u @ s
    return B, L


def plotBasRelief(B, mu, nu, lam):

    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter
    
    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """
    print("Parameters mu , nu , lam ", mu , nu , lam)
    G = np.array([[1,0,0],[0,1,0],[mu,nu,lam]])
    trans = np.linalg.inv(G).T
    newB =  B
    albedos, normals = estimateAlbedosNormals(newB)
    Nt = enforceIntegrability(normals, s)
    Nt = trans @ Nt
    surface = estimateShape(Nt, s)
    plotSurface(surface)
    print("ENDDDDDD")


    if __name__ == "__main__":

        # Part 2 (b)
        # Your code here
        I, l, s = loadData()
        print("Original L ",l)
        B,L  = estimatePseudonormalsUncalibrated(I)

        print("New L ", L.T)
        albedos, normals = estimateAlbedosNormals(B)
        albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

        # Part 2 (d)
        surface = estimateShape(normals, s)
        plotSurface(surface)

        # Part 2 (e)
        Nt = enforceIntegrability(normals, s)
        surface = estimateShape(Nt, s)
        plotSurface(surface)
        print("ABLATION")
        # Part 2 (f)
        params = np.array([[1,0,1],
                        [10,0,1],
                        [0,1,1],
                        [0,10,1],
                        [0,0,0.1],
                        [0,0,10]])

        # params = np.array([[0,5,1],
        #                   [0,10,1],])


        for i in range(params.shape[0]):
            plotBasRelief(B, params[i,0], params[i,1], params[i,2],s)
            
