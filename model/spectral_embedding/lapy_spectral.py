import numpy as np
import scipy.sparse.linalg as sla
import torch

from model.spectral_embedding.lapy import TriaMesh, Solver


def grid_spectral_meshes(H, W, k):
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

    mesh = TriaMesh(points, faces)
    fem = Solver(mesh)
    evals, LBO_MATRIX = fem.eigs(k=k)

    return torch.from_numpy(LBO_MATRIX).float()
