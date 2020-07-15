import torch


def mse(im1, im2):
    '''
    MSE between two images.
    '''

    return torch.mean((im1 - im2) ** 2)

def psnr_from_mse(v):
    '''
    Convert MSE to PSNR.
    '''

    return -10.0 * (torch.log(v) / torch.log(torch.tensor([10.0])))
