import torch
import numpy as np
import os
import imageio
from training import metrics
from tqdm import tqdm, trange
from utils.ray import generate_rays


def test_net(img_H, img_W, focal, renderer, test_poses, on_progress=None, on_complete=None):
    '''
    Test the network and generate results.

    Arguments:
        img_H: height of image plane.
        img_W: width of image plane.
        focal: focal length.
        renderer: the volume renderer.
        test_poses: poses used to test the network. (#poses, 4, 4)
        on_progress: a callback function invoked per generation of a result.
        on_complete: a callback function invoked after generating all results.
    '''

    rgb_maps = []
    
    with torch.no_grad():
        with tqdm(test_poses) as pbar: 
            for j, test_pose in enumerate(pbar):
                # Generate rays for all pixels
                ray_oris, ray_dirs = generate_rays(img_H, img_W, focal, test_pose)
                ray_oris = torch.reshape(ray_oris, [-1, 3])
                ray_dirs = torch.reshape(ray_dirs, [-1, 3])
                test_batch_rays = torch.stack([ray_oris, ray_dirs], dim=0)

                # Retrieve testing results
                rgb_map, _ = renderer(test_batch_rays)
                rgb_maps.append(torch.reshape(rgb_map, [img_H, img_W, 3]).cpu().numpy())

                # Handle each testing result
                if on_progress:
                    on_progress(j, rgb_maps[-1])
        
        # Handle all testing results
        if on_complete:
            on_complete(np.stack(rgb_maps, 0))
