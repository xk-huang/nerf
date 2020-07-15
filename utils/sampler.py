import torch
import numpy as np

def sample_grid_2d(H, W, N):
    '''
    Sample cells in an H x W mesh grid.

    Arguments:
        H: height of the mesh grid.
        W: width of the mesh grid.
        N: the number of samples.

    Returns:
        A tuple: (sampled rows, sampled columns).
    '''

    if N > W * H:
        N = W * H

    # Create a 2D mesh grid where each element is the coordinate of the cell
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), dim=-1)  # (H, W, 2)
    # Flat the mesh grid
    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
    # Sample N cells in the mesh grid
    select_inds = np.random.choice(coords.shape[0], size=[N], replace=False)  # (N, 2)
    # Sample N cells among the mesh grid
    select_coords = coords[select_inds].long()  # (N, 2)

    return select_coords[:, 0], select_coords[:, 1]  # (N), (N)


def sample_along_rays(near, far, N_samples, lindisp=False, perturb=True):
    '''
    Sample points along rays

    Arguments:
        near: a vector containing nearest point for each ray. (N_rays)
        far: a vector containing furthest point for each ray. (N_rays)
        N_samples: the number of sampled points for each ray.
        lindisp: True for sample linearly in inverse depth rather than in depth (used for some datasets).
        perturb: True for stratified sampling. False for uniform sampling.
    
    Returns:
        Samples where j-th component of the i-th row is the j-th sampled position along the i-th ray.
    '''

    # The number of rays
    N_rays = near.shape[0]

    # Uniform samples along rays
    t_vals = torch.linspace(0., 1., steps=N_samples)  # (N_samples)
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)

    z_vals = z_vals.expand([N_rays, N_samples])  # (N_rays, N_samples)

    if perturb:
        # Get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        # Stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    return z_vals  # (N_rays, N_samples)
