import torch
from training import metrics


def train_net(config, iter, renderer, optimizer, rays, gt):
    '''
    Train a network.

    Arguments:
        config: configuration.
        iter: current iterations.
        renderer: a volume renderer.
        optimizer: a network optimizer.
        rays: a batch of rays for training. (#rays * #samples, 6)
        gt: the groundtruth.

    Returns:
        A tuple: (MSE loss, PSNR).
    '''

    rgb_map_fine, rgb_map_coarse = renderer(rays)

    optimizer.zero_grad()
    loss = metrics.mse(rgb_map_fine, gt)
    psnr = metrics.psnr_from_mse(loss)
    loss = loss + metrics.mse(rgb_map_coarse, gt)

    # Back-propagation
    loss.backward()
    # Perform a single optimization step
    optimizer.step()

    # Update learning rate
    decay_rate = 0.1
    decay_steps = config.lrate_decay * 1000
    new_lrate = config.lrate * (decay_rate ** (iter / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate

    return loss, psnr
