import os
import imageio
import numpy as np

# Convert normalized color to 8-bit color
to8b = lambda x: (255 * np.clip(x, 0.0, 1.0)).astype(np.uint8)


def save_image(j, rgb_img, save_dir):
    """
    Save images of the j-th result in a specified directory.

    Arguments:
        j: the image index.
        rgb_img: color map. (H, W)
        save_dir: the path to save the result.
    """

    file_path = os.path.join(save_dir, "{:04d}.png".format(j))
    imageio.imwrite(file_path, to8b(rgb_img))


def save_video(i, rgb_imgs, save_dir, fps=30):
    """
    Save a video of the i-th iteration in a specified directory.

    Arguments:
        j: the image index.
        rgb_imgs: a sequence of color maps. (#imgs, H, W)
        save_dir: the path to save the result.
    """

    file_path = os.path.join(save_dir, "test_{:04d}_rgb.mp4".format(i))
    imageio.mimwrite(file_path, to8b(rgb_imgs), fps=fps, quality=8)
