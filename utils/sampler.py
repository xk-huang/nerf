import torch
import numpy as np

# from torchsearchsorted import searchsorted
from torch import searchsorted


def sample_grid_2d(H, W, N):
    """
    Sample cells in an H x W mesh grid.

    Arguments:
        H: height of the mesh grid.
        W: width of the mesh grid.
        N: the number of samples.

    Returns:
        A tuple: (sampled rows, sampled columns).
    """

    if N > W * H:
        N = W * H

    # Create a 2D mesh grid where each element is the coordinate of the cell
    coords = torch.stack(
        torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), dim=-1
    )  # (H, W, 2)
    # Flat the mesh grid

    coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
    # Sample N cells in the mesh grid
    select_inds = np.random.choice(coords.shape[0], size=[N], replace=False)  # (N, 2)
    # Sample N cells among the mesh grid
    select_coords = coords[select_inds].long()  # (N, 2)

    return select_coords[:, 0], select_coords[:, 1]  # (N), (N)


def sample_along_rays(near, far, N_samples, lindisp=False, perturb=True):
    """
    Sample points along rays

    Arguments:
        near: a vector containing nearest point for each ray. (N_rays)
        far: a vector containing furthest point for each ray. (N_rays)
        N_samples: the number of sampled points for each ray.
        lindisp: True for sample linearly in inverse depth rather than in depth (used for some datasets).
        perturb: True for stratified sampling. False for uniform sampling.

    Returns:
        Samples where j-th component of the i-th row is the j-th sampled position along the i-th ray.
    """

    # The number of rays
    N_rays = near.shape[0]

    # Uniform samples along rays
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples)  # (N_samples)
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


def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous()
    inds = searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
