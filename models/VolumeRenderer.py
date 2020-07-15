import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsearchsorted import searchsorted
from utils.sampler import sample_along_rays


def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous()
    inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


class VolumeRenderer(nn.Module):
    def __init__(self, config, model_coarse, model_fine, embedder_p, embedder_d, near=0, far=1e6):
        
        super(VolumeRenderer, self).__init__()

        # self.kwargs = kwargs
        
        self.config = config
        # coarse model
        self.model_coarse = model_coarse
        # fine model
        self.model_fine = model_fine
        # embedder for positions
        self.embedder_p = embedder_p
        # embedder for view-in directions
        self.embedder_d = embedder_d

        self.near = near
        self.far = far

    def forward(self, rays):
        # make the number of rays be multiple of the chunk size
        N_rays = (rays.shape[1] // self.config.chunk + 1) * self.config.chunk
        N = self.config.N_samples

        res_ls = {
            'rgb_map_coarse': [],
            'rgb_map_fine': []
        }

        for i in range(0, N_rays, self.config.chunk):
            ray_oris, ray_dirs = rays[:, i:i + self.config.chunk, :]
            view_dirs = torch.reshape(ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True), [-1, 3]).float()
            near, far = self.near * torch.ones_like(ray_dirs[..., :1]), self.far * torch.ones_like(ray_dirs[..., :1])
            M = ray_oris.shape[0]  # chunk size
            if M == 0:
                continue

            '''
            stratified sampling along rays
            '''
            # # length of sample steps
            # step_len = (self.far - self.near) / N
            # # uniform samples among near and far
            # u_samples = torch.linspace(self.near, self.far, N + 1).expand([M, N + 1])
            # # jitter each uniform sample to generate stratified samples
            # s_samples = (u_samples + torch.rand(u_samples.shape) * step_len)[:, :-1]  # (M, N)

            s_samples = sample_along_rays(near, far, N)


            # position samples along rays
            pos_samples = ray_oris.unsqueeze(1) + ray_dirs.unsqueeze(1) * s_samples.unsqueeze(2)  # (M, N, 3)
            # expand ray directions to the same shape of samples
            dir_samples = view_dirs.unsqueeze(1).expand(pos_samples.shape)

            pos_samples = torch.reshape(pos_samples,  [-1, 3])  # (M * N, 3)
            dir_samples = torch.reshape(dir_samples, [-1 ,3])  # (M * N, 3)


            # '''
            # position encoding
            # '''
            # h_ls = []
            # if self.embedder_p:
            #     pos_samples = self.embedder_p(pos_samples)
            #     h_ls.append(pos_samples)
            # if self.embedder_d:
            #     dir_samples = self.embedder_d(dir_samples)
            #     h_ls.append(dir_samples)
            
            # '''
            # get result from coarse model
            # '''
            # # make the input of the coarse model
            # h = torch.cat(h_ls, dim=-1)
            # # retrieve optic data of the sampled points from the coarse model
            # optic_d = self.model_coarse(h)
            # optic_d = torch.reshape(optic_d, [M, N, 4])

            # retrieve optic data from the network
            optic_d = self._run_network(pos_samples, dir_samples, self.model_coarse)
            optic_d = torch.reshape(optic_d, [M, N, 4])
            
            # composite optic data to generate a RGB image
            rgb_map_coarse, weights_coarse = self._composite(optic_d, s_samples, ray_dirs)

            if self.config.N_importance > 0:
                z_vals_mid = .5 * (s_samples[...,1:] + s_samples[...,:-1])
                z_samples = sample_pdf(z_vals_mid, weights_coarse[...,1:-1],self.config.N_importance)
                z_samples = z_samples.detach()

                z_vals, _ = torch.sort(torch.cat([s_samples, z_samples], -1), -1)
                pts = ray_oris[...,None,:] + ray_dirs[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

                dir_samples = view_dirs.unsqueeze(1).expand(pts.shape)

                pts = torch.reshape(pts, [-1, 3])
                dir_samples = torch.reshape(dir_samples, [-1 ,3])

                optic_d = self._run_network(pts, dir_samples, self.model_fine)
                optic_d = torch.reshape(optic_d, [M, N + self.config.N_importance, 4])

                rgb_map_fine, _ = self._composite(optic_d, z_vals, ray_dirs)
            else:
                rgb_map_fine = rgb_map_coarse

            # rgb_map_fine = rgb_map_coarse

            res_ls['rgb_map_coarse'].append(rgb_map_coarse)
            res_ls['rgb_map_fine'].append(rgb_map_fine)

        res = {k : torch.cat(res_ls[k], 0) for k in res_ls}
        
        return res['rgb_map_fine'], res['rgb_map_coarse']

    # def _composite(self, s_samples, vol_info):
    #     den = vol_info[..., 3]
    #     color = vol_info[..., :3]

    #     # retrieve length of sample steps
    #     lens = s_samples[..., 1:] - s_samples[..., :-1]
    #     # length of the last sample step
    #     last_len = torch.tensor([1e6]).expand(lens[..., :1].shape)
    #     # full length of the sample steps
    #     lens = torch.cat([lens, last_len], dim=-1)
    #     # alpha values
    #     alpha = 1.0 - torch.exp(den * lens)
    #     # weights
    #     w_s = alpha * torch.cumprod(torch.cat([torch.ones([alpha.shape[0], 1]), 1.0 - alpha], dim=-1), dim=-1)[..., :-1]
    #     # RGB
    #     rgb = torch.sum(w_s.unsqueeze(-1) * color, dim=-2)

    #     if self.config.white_bkgd:
    #         white_bg = 1.0 - torch.sum(w_s, dim=-1)
    #         rgb += white_bg.unsqueeze(-1)

    #     return rgb, w_s

    def _run_network(self, pts, view_dirs, model):
        # '''
        # position encoding
        # '''
        # h_ls = []
        # if self.embedder_p:
        #     pos_samples = self.embedder_p(pos_samples)
        #     h_ls.append(pos_samples)
        # if self.embedder_d:
        #     dir_samples = self.embedder_d(dir_samples)
        #     h_ls.append(dir_samples)
        
        # '''
        # get result from coarse model
        # '''
        # # make the input of the coarse model
        # h = torch.cat(h_ls, dim=-1)
        # # retrieve optic data of the sampled points from the coarse model
        # optic_d = self.model_coarse(h)

        # return optic_d

        # return self.kwargs['network_query_fn'](pts, view_dirs, model)
        
        # inputs_flat = torch.reshape(pts, [-1, pts.shape[-1]])
        inputs_flat = pts
        embedded = self.embedder_p(inputs_flat)

        if view_dirs is not None:
            # input_dirs = view_dirs[:,None].expand(pts.shape)
            # input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            input_dirs_flat = view_dirs
            embedded_dirs = self.embedder_d(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        # outputs_flat = batchify(fn, netchunk)(embedded)
        chunk = self.config.netchunk
        outputs_flat = torch.cat([model(embedded[i:i+chunk]) for i in range(0, embedded.shape[0], chunk)], 0)
        # outputs = torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs_flat

    def _transfer(self, optic_d, dists):
        rgbs = torch.sigmoid(optic_d[..., :3])  # (chunk_size, N_samples, 3)
        alphas = 1.0 - torch.exp(-F.relu(optic_d[..., 3]) * dists)  # (chunk_size, N_samples)

        return rgbs, alphas

    def _composite(self, optic_d, s_samples, rays_d):
        # raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

        # distances between each samples
        dists = s_samples[..., 1:] - s_samples[..., :-1]  # (chunk_size, N_samples - 1)
        dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[...,:1].shape)], -1)  # (chunk_size, N_samples)

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        # rgb = torch.sigmoid(optic_d[...,:3])  # [chunk_size, N_samples, 3]
        # noise = 0.
        # alpha = raw2alpha(optic_d[...,3] + noise, dists)  # [chunk_size, N_samples]

        # retrieve display colors and alphas for each samples by a transfer function
        rgbs, alphas = self._transfer(optic_d, dists)

        weights = alphas * torch.cumprod(torch.cat([torch.ones((alphas.shape[0], 1)), 1.0 - alphas + 1e-10], dim=-1)[:, :-1], dim=-1)  # (chunk_size, N_samples)
        rgb_map = torch.sum(weights[..., None] * rgbs, dim=-2)  # (chunk_size, 3)
        acc_map = torch.sum(weights, -1)  # (chunk_size)

        if self.config.white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        return rgb_map, weights
