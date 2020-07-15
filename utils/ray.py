import torch
import numpy as np


def generate_rays(h, w, f, pose):
    '''
    Given an image plane, generate rays from the camera origin to each pixel on the image plane.

    Arguments:
        h: height of the image plane.
        w: width of the image plane.
        f: focal length of the image plane.
        pose: the extrinsic parameters of the camera. (3, 4) or (4, 4)

    Returns:
        A tuple: origins of rays, directions of rays
    '''

    # Coordinates of the 2D grid
    cols = (torch.linspace(-w / 2, w - 1 - w / 2, w) / f).repeat([h, 1])  # (h, w)
    rows = (-torch.linspace(-h / 2, h - 1 - h / 2, h) / f).unsqueeze(1).repeat([1, w])  # (h, w)

    # Ray directions for all pixels
    ray_dirs = torch.stack([cols, rows, -torch.ones_like(cols)], dim=-1)  # (h, w, 3)
    # Apply rotation transformation to make each ray orient according to the camera
    ray_dirs = torch.sum(ray_dirs.unsqueeze(2) * pose[:3, :3], dim=-1)
    # Origin position
    rays_oris = pose[:3,-1].expand_as(ray_dirs)  # (h, w, 3)

    return rays_oris.float(), ray_dirs.float()  # (h, w, 3), (h, w, 3)
