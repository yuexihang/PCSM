import robust_laplacian
import numpy as np
import scipy.sparse.linalg as sla
import torch

def onedim_spectral_meshes(L, k, which='LM', return_values = False):
    points = []
    for r in range(L):
        points.append( [ r / L, 0, 0 ] )
    points = np.array(points)
    
    faces = []
    IDX = np.zeros( shape=(L), dtype=int )
    for r in range( L ):
        IDX[r] = r
    for r in range(L):
        if r + 2 < L:
            faces.append( [ IDX[r], IDX[r+1] , IDX[r+2] ] )
            faces.append( [ IDX[r], IDX[r+2] , IDX[r+1] ] )
    faces = np.array(faces)

    L, M = robust_laplacian.mesh_laplacian(points, faces)
    evals, evecs = sla.eigsh(L, k, M, sigma=1e-8, which=which)
    if return_values: return torch.from_numpy(evals).float(), torch.from_numpy(evecs).float()
    return torch.from_numpy(evecs).float()

def grid_spectral_points(H, W, k, which='LM', n_neighbors=30, return_values = False):
    # prepare points V * 3
    points = []
    for r in range(H):
        for c in range(W):
            points.append( [ r / H, c / W, 0 ] )
    points = np.array(points)
    L, M = robust_laplacian.point_cloud_laplacian(points, n_neighbors=n_neighbors)
    evals, evecs = sla.eigsh(L, k, M, sigma=1e-8, which=which)
    if return_values: return torch.from_numpy(evals).float(), torch.from_numpy(evecs).float()
    return torch.from_numpy(evecs).float()

def grid_spectral_meshes(H, W, k, which='LM'):
    # prepare points V * 3
    points = []
    for r in range(H):
        for c in range(W):
            points.append( [ r / H, c / W, 0 ] )
    points = np.array(points)
    # prepare faces F * 3
    faces = []
    IDX = np.zeros( shape=(H,W), dtype=int )
    for r in range( H ):
        for c in range(W):
            IDX[r,c] = r * W + c
    for r in range(H):
        for c in range(W):
            if r + 1 < H and c + 1 < W:
                faces.append( [ IDX[r,c], IDX[r,c+1]  , IDX[r+1,c+1] ] )
                faces.append( [ IDX[r,c], IDX[r+1,c+1], IDX[r+1,  c] ] )
    faces = np.array(faces)

    L, M = robust_laplacian.mesh_laplacian(points, faces)
    evals, evecs = sla.eigsh(L, k, M, sigma=1e-8, which=which)
    return torch.from_numpy(evecs).float()


def points_spectral_2d( points, n_neighbors = 30 , k = 100, return_values = False):
    points = np.concatenate( [points, np.zeros( (points.shape[0], 1) ) ], axis=1 ) # N, 3
    L, M = robust_laplacian.point_cloud_laplacian(points, n_neighbors=n_neighbors)
    evals, evecs = sla.eigsh(L, k, M, sigma=1e-8)
    if return_values: return torch.from_numpy(evals).float(), torch.from_numpy(evecs).float()
    return torch.from_numpy(evecs).float()

def meshes_spectral_2d( points, faces, which='LM', k = 100, return_values = False):
    L, M = robust_laplacian.mesh_laplacian(points, faces)
    evals, evecs = sla.eigsh(L, k, M, sigma=1e-8, which=which)
    if return_values: return torch.from_numpy(evals).float(), torch.from_numpy(evecs).float()
    return torch.from_numpy(evecs).float()
