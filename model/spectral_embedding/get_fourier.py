import numpy as np
from scipy import sparse as sp
from math import sqrt
import torch

def face_to_edges(faces):
    edges = []
    for cell in faces:
        edges.append( ( cell[0], cell[1] ) )
        edges.append( ( cell[1], cell[2] ) )
        edges.append( ( cell[2], cell[0] ) )
    edges = np.array( edges )
    return edges

def face4_to_edges(faces):
    edges = []
    for cell in faces:
        edges.append( ( cell[0], cell[1] ) )
        edges.append( ( cell[1], cell[2] ) )
        edges.append( ( cell[2], cell[0] ) )
        edges.append( ( cell[0], cell[3] ) )
        edges.append( ( cell[1], cell[3] ) )
        edges.append( ( cell[2], cell[3] ) )
    edges = np.array( edges )
    return edges

def edges_to_adj(edges, points, t = 1, use_distance = False):
    n = points.shape[0]

    if use_distance:
        distance = []
        for eid in range( edges.shape[0] ):
            p1, p2 = points[ edges[eid][0] ], points[ edges[eid][1] ] # (2) (2)
            dis = ( (p1 - p2) ** 2 ).sum()
            distance.append( np.exp( - ( dis / t ) ) )
        distance = np.array( distance )
    else :
        distance = np.ones( edges.shape[0] )

    a = sp.csr_matrix( ( distance, (edges[:, 0], edges[:, 1])), shape=(n, n) )
    a = a + a.T

    return a


def degree_matrix(A):
    degrees = np.array(A.sum(1)).flatten()
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def laplacian_new(A):
    deg = degree_matrix(A)
    return deg, deg - A

def get_fourier_new(adj, k=100, which="LM"):
    deg, lap = laplacian_new(adj)
    _, u = sp.linalg.eigsh(lap, k=k, M=deg , which=which)
    n = lap.shape[0]
    u *= np.sqrt(n)

    return u

def grid2spectral(grid_shape, k = 20, which="LM"):
    xlen, ylen = grid_shape
    point_num = xlen * ylen

    IDX = np.zeros( shape=grid_shape, dtype=int )
    for r in range( xlen ):
        for c in range(ylen):
            IDX[r,c] = r * ylen + c

    edges = []
    for r in range( xlen ):
        for c in range( ylen ):
            nextr = (r+1) % xlen
            if c + 1 < ylen:
                edges.append( [ IDX[r,c], IDX[ r, c+1 ] ] )
            if r + 1 < xlen:
                edges.append( [ IDX[r,c], IDX[ r+1, c ]  ] )
    edges = np.array( edges )


    adj = sp.csr_matrix( (np.ones(edges.shape[:1]), (edges[:, 0], edges[:, 1])), shape=(point_num, point_num) )
    adj = adj + adj.T
    adj.data[:] = 1.0 
    
    u = np.array( get_fourier_new(adj, k = k,  which=which) ) # point num * K
    return torch.tensor( u, dtype=torch.float ).reshape( xlen, ylen, -1 ) # x, y, k