import os
import numpy as np
import imageio
import time
import configargparse
import torch

# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.load_blender import load_blender_data
from models import VolumeRenderer, NeRF
from training import train_net, test_net
from utils import Embedder
from utils.sampler import sample_grid_2d
from utils.ray import generate_rays
from utils.results_handler import save_image, save_video


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
        'include_input' : True,
        'input_dims' : 3,
        'max_freq_pow' : multires-1,
        'num_freqs' : multires,
        'log_sampling' : True,
        'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo(x)
    return embed, embedder_obj.out_dims


def create_nerf(config, device, out_dir):
    embed_fn, input_ch = get_embedder(config.multires, config.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if config.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(config.multires_views, config.i_embed)
    
    '''
    Create networks
    '''
    output_ch = 4
    skips = [4]
    model_coarse = NeRF(D=config.netdepth, W=config.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=config.use_viewdirs).to(device)
    grad_vars = list(model_coarse.parameters())

    model_fine = None
    if config.N_importance > 0:
        model_fine = NeRF(D=config.netdepth_fine, W=config.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=config.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    '''
    Create an optimizer
    '''
    optimizer = torch.optim.Adam(params=grad_vars, lr=config.lrate, betas=(0.9, 0.999))

    '''
    Load checkpoints
    '''
    start_iter = 0
    ckpts = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if '.tar' in f]
    
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not config.no_reload:
        # Reload the latest ckpt
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        # Load training steps
        start_iter = ckpt['global_steps'] + 1

        # Load optimizer states
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load network weights
        model_coarse.load_state_dict(ckpt['network_coarse_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    else:
       print('No ckpt reloaded') 

    return start_iter, optimizer, model_coarse, model_fine, embed_fn, embeddirs_fn


def train_pipeline(config, device, out_dir):


    '''
    Load data
    '''

    # if config.dataset_type == 'llff':
    #     images, poses, bds, render_poses, i_test = load_llff_data(config.datadir, config.factor,
    #                                                               recenter=True, bd_factor=.75,
    #                                                               spherify=config.spherify)
    #     hwf = poses[0,:3,-1]
    #     poses = poses[:,:3,:4]
    #     print('Loaded llff', images.shape, render_poses.shape, hwf, config.datadir)
    #     if not isinstance(i_test, list):
    #         i_test = [i_test]

    #     if config.llffhold > 0:
    #         print('Auto LLFF holdout,', config.llffhold)
    #         i_test = np.arange(images.shape[0])[::config.llffhold]

    #     i_val = i_test
    #     i_train = np.array([i for i in np.arange(int(images.shape[0])) if
    #                     (i not in i_test and i not in i_val)])

    #     print('DEFINING BOUNDS')
    #     if config.no_ndc:
    #         near = torch.min(bds) * .9
    #         far = torch.max(bds) * 1.
    #     else:
    #         near = 0.
    #         far = 1.
    #     print('NEAR FAR', near, far)

    # elif config.dataset_type == 'blender':
    #     images, poses, render_poses, hwf, i_split = load_blender_data(config.datadir, config.half_res, config.testskip)
    #     print('Loaded blender', images.shape, render_poses.shape, hwf, config.datadir)
    #     i_train, i_val, i_test = i_split

    #     near = 2.
    #     far = 6.

    #     if config.white_bkgd:
    #         images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    #     else:
    #         images = images[...,:3]

    # elif config.dataset_type == 'NHR':
    #     images, poses, render_poses, hwf, i_split, mask_nhr = IBRay_NHR(config.datadir)
    #     print('Loaded NHR', images.shape, render_poses.shape, hwf, config.datadir)
    #     i_train, i_val, i_test = i_split

    #     near = 1.
    #     far = 5.5

    # elif config.dataset_type == 'NOH':
    #     images, poses, render_poses, hwf, i_split, mask_nhr = IBRay_NOH(config.datadir)
    #     print('Loaded NOH', images.shape, render_poses.shape, hwf, config.datadir)
    #     i_train, i_val, i_test = i_split

    #     near = 10
    #     far = 40

    # elif config.dataset_type == 'deepvoxels':

    #     images, poses, render_poses, hwf, i_split = load_dv_data(scene=config.shape,
    #                                                              basedir=config.datadir,
    #                                                              testskip=config.testskip)

    #     print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, config.datadir)
    #     i_train, i_val, i_test = i_split

    #     hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
    #     near = hemi_R-1.
    #     far = hemi_R+1.

    print('>>> Loading dataset')

    if config.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(config.datadir, config.half_res, config.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, config.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if config.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
    else:
        print('Unknown dataset type', config.dataset_type, 'exiting')
        return

    if config.render_test:
        render_poses = np.array(poses[i_test])

    print('TRAIN views:', i_train)
    print('TEST views:', i_test)
    print('VAL views:', i_val)

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]


    '''
    Setup logging and directory for results
    '''
    print('>>> Saving checkpoints and results in', out_dir)
    # Create output directory if not existing
    os.makedirs(out_dir, exist_ok=True)
    # Record current configuration 
    with open(os.path.join(out_dir, 'configs.txt'), 'w+') as config_f:
        attrs = vars(config)
        for k in attrs:
            config_f.write('%s = %s\n' % (k, attrs[k]))
    
    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))


    '''
    Create network models, optimizer and renderer
    '''
    print('>>> Creating models')

    # Create nerf model
    start_iter, optimizer, model_coarse, model_fine, embed_fn, embeddirs_fn  = create_nerf(config, device, out_dir)
    # Training steps
    global_steps = start_iter
    # Create volume renderer
    renderer = VolumeRenderer(config, model_coarse, model_fine, embed_fn, embeddirs_fn, near, far)


    '''
    Only render results by pre-trained models
    '''

    if config.render_only:
        print('>>> Render only')

        # Move testing data to GPU
        render_poses = torch.Tensor(render_poses).to(device)

        # Path to save rendering results
        render_save_dir = os.path.join(out_dir, 'renderonly_{:06d}'.format(global_steps))
        os.makedirs(render_save_dir, exist_ok=True)

        save_img_fn = lambda j, img: save_image(j, img, render_save_dir)
        save_video_fn = lambda imgs: save_video(global_steps, imgs, render_save_dir)

        print('Rendering (iter=%s):' % global_steps)
        
        test_net(H, W, focal, renderer, render_poses, None, on_progress=save_img_fn, on_complete=save_video_fn)

        return


    '''
    Start training
    '''

    print('>>> Start training')

    N_rand = config.N_rand

    # Prepare ray batch tensor if batching random rays
    use_batching = not config.no_batching
    use_batching = False
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')

        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]

        if config.dataset_type == 'NHR':
            mask_nhr = np.stack([mask_nhr[i] for i in i_train], 0) # train images only
            mask_nhr =  np.reshape(mask_nhr, [-1])
            print(mask_nhr.shape)
            print(rays_rgb.shape)
            rays_rgb = rays_rgb[mask_nhr>0.8,:,:]
        
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    # Maximum training iterations
    N_iters = config.N_iters
    if start_iter >= N_iters:
        return

    with tqdm(range(1, N_iters + 1)) as pbar:
        pbar.n = start_iter

        for i in pbar:
            # Show progress
            pbar.set_description('Iter %d' % (global_steps + 1))
            pbar.update()

            # Start time of the current iteration
            time0 = time.time()

            # Sample random ray batch
            if use_batching:
                # Random over all images
                batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
                batch = torch.transpose(batch, 0, 1)
                batch_rays, target_s = batch[:2], batch[2]

                i_batch += N_rand
                if i_batch >= rays_rgb.shape[0]:
                    pbar.write("Shuffle data after an epoch!")
                    rand_idx = torch.randperm(rays_rgb.shape[0])
                    rays_rgb = rays_rgb[rand_idx]
                    i_batch = 0

            # Random sampling from one image
            else:
                img_i = np.random.choice(i_train)
                target = images[img_i]
                pose = poses[img_i, :3,:4]

                if N_rand is not None:
                    rays_o, rays_d = generate_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
                    sampled_rows, sampled_cols = sample_grid_2d(H, W, N_rand)
                    rays_o = rays_o[sampled_rows, sampled_cols]  # (N_rand, 3)
                    rays_d = rays_d[sampled_rows, sampled_cols]  # (N_rand, 3)
                    batch_rays = torch.stack([rays_o, rays_d], 0)
                    target_s = target[sampled_rows, sampled_cols]  # (N_rand, 3)


            loss, psnr = train_net(config, global_steps, renderer, optimizer, batch_rays, target_s)
            
            pbar.set_postfix(time=time.time() - time0, loss=loss.item(), psnr=psnr.item())


            '''
            Logging
            '''

            # Save training states
            if (global_steps + 1) % config.i_ckpt == 0:
                path = os.path.join(out_dir, '{:06d}.tar'.format((global_steps + 1)))
                torch.save({
                    'global_steps': global_steps,
                    'network_coarse_state_dict': model_coarse.state_dict(),
                    'network_fine_state_dict': model_fine.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                pbar.write('Saved checkpoints at', path)

            # Save testing results
            if (global_steps + 1) % config.i_testset == 0:
                test_save_dir = os.path.join(out_dir, 'test_{:06d}'.format(global_steps + 1))
                os.makedirs(test_save_dir, exist_ok=True)

                save_img_fn = lambda j, img: save_image(j, img, test_save_dir)
                save_video_fn = lambda imgs: save_video(global_steps + 1, imgs, test_save_dir)

                pbar.write('Testing (iter=%s):' % (global_steps + 1))

                test_time, test_loss, test_psnr = test_net(H, W, focal, renderer, torch.Tensor(poses[i_test]).to(device), images[i_test],
                            on_progress=save_img_fn, on_complete=save_video_fn)
                
                pbar.write('Testing results: [ Mean Time: %.4fs, Loss: %.4f, PSNR: %.4f ]' % (test_time, test_loss, test_psnr))

            """
            if i%config.i_print==0 or i < 10:

                print(expname, i, psnr.numpy(), loss.numpy(), global_steps.numpy())
                print('iter time {:.05f}'.format(dt))
                with tf.contrib.summary.record_summaries_every_n_global_steps(config.i_print):
                    tf.contrib.summary.scalar('loss', loss)
                    tf.contrib.summary.scalar('psnr', psnr)
                    tf.contrib.summary.histogram('tran', trans)
                    if config.N_importance > 0:
                        tf.contrib.summary.scalar('psnr0', psnr0)


                if i%config.i_img==0:

                    # Log a rendered validation view to Tensorboard
                    img_i=np.random.choice(i_val)
                    target = images[img_i]
                    pose = poses[img_i, :3,:4]
                    with torch.no_grad():
                        rgb, disp, acc, extras = render(H, W, focal, chunk=config.chunk, c2w=pose,
                                                            **render_kwargs_test)

                    psnr = mse2psnr(img2mse(rgb, target))

                    with tf.contrib.summary.record_summaries_every_n_global_steps(config.i_img):

                        tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                        tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                        tf.contrib.summary.scalar('psnr_holdout', psnr)
                        tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                    if config.N_importance > 0:

                        with tf.contrib.summary.record_summaries_every_n_global_steps(config.i_img):
                            tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                            tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                            tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
            """

            global_steps += 1

def cuda_setup(id):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(id)

    return device

def main():
    parser = configargparse.ArgumentParser()
    # General options
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--name", required=True, type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./results/', help='where to store ckpts and results')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')
    # Training options
    parser.add_argument("--N_iters", type=int, default=200000, help='max training iterations')
    parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--gpu",   type=int, default=0, help='the index of gpu device')
    # Rendering options
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0, help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_only", action='store_true', help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    # Dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    # DeepVoxels flags
    parser.add_argument("--shape", type=str, default='greek', help='options : armchair / cube / greek / vase')
    # Blender flags
    parser.add_argument("--white_bkgd", action='store_true', help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', help='load blender synthetic data at 400x400 instead of 800x800')
    # LLFF flags
    parser.add_argument("--factor", type=int, default=8, help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, help='will take every 1/N images as LLFF test set, paper uses 8')
    # Logging/saving options
    parser.add_argument("--i_print", type=int, default=100, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500, help='frequency of tensorboard image logging')
    parser.add_argument("--i_ckpt", type=int, default=10000, help='frequency of ckpt saving')
    parser.add_argument("--i_testset", type=int, default=20000, help='frequency of testset saving')

    config = parser.parse_args()

    # Cuda device
    device = cuda_setup(config.gpu)
    
    # Output directory
    basedir = config.basedir
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    # Experiment name
    exp_name = config.dataset_type + "_" + config.name
    # get the experiment number
    exp_num = max([int(fn.split('_')[-1]) for fn in os.listdir(basedir) if fn.find(exp_name) >= 0] + [0])
    if config.no_reload:
        exp_num += 1
    
    # Output directory
    out_dir = os.path.join(basedir, exp_name + "_"+ str(exp_num))

    # Start training pipeline
    train_pipeline(config, device, out_dir)

if __name__ == '__main__':
    main()
