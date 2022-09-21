import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sampler import sample_along_rays, sample_pdf


class VolumeRenderer(nn.Module):
    def __init__(
        self, config, model_coarse, model_fine, embedder_p, embedder_d, near=0, far=1e6
    ):
        super(VolumeRenderer, self).__init__()

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

        res_ls = {"rgb_map_coarse": [], "rgb_map_fine": []}

        for i in range(0, N_rays, self.config.chunk):
            ray_oris, ray_dirs = rays[:, i : i + self.config.chunk, :]
            view_dirs = torch.reshape(
                ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True), [-1, 3]
            ).float()
            near, far = self.near * torch.ones_like(
                ray_dirs[..., :1]
            ), self.far * torch.ones_like(ray_dirs[..., :1])
            M = ray_oris.shape[0]  # chunk size
            if M == 0:
                continue

            """
            stratified sampling along rays
            """
            s_samples = sample_along_rays(near, far, N)

            # position samples along rays
            pos_samples = ray_oris.unsqueeze(1) + ray_dirs.unsqueeze(
                1
            ) * s_samples.unsqueeze(
                2
            )  # (M, N, 3)
            # expand ray directions to the same shape of samples
            dir_samples = view_dirs.unsqueeze(1).expand(pos_samples.shape)

            pos_samples = torch.reshape(pos_samples, [-1, 3])  # (M * N, 3)
            dir_samples = torch.reshape(dir_samples, [-1, 3])  # (M * N, 3)

            # retrieve optic data from the network
            optic_d = self._run_network(pos_samples, dir_samples, self.model_coarse)
            optic_d = torch.reshape(optic_d, [M, N, 4])

            # composite optic data to generate a RGB image
            rgb_map_coarse, weights_coarse = self._composite(
                optic_d, s_samples, ray_dirs
            )

            if self.config.N_importance > 0:
                z_vals_mid = 0.5 * (s_samples[..., 1:] + s_samples[..., :-1])
                z_samples = sample_pdf(
                    z_vals_mid, weights_coarse[..., 1:-1], self.config.N_importance
                )
                z_samples = z_samples.detach()

                z_vals, _ = torch.sort(torch.cat([s_samples, z_samples], -1), -1)
                pts = (
                    ray_oris[..., None, :]
                    + ray_dirs[..., None, :] * z_vals[..., :, None]
                )  # [N_rays, N_samples + N_importance, 3]

                dir_samples = view_dirs.unsqueeze(1).expand(pts.shape)

                pts = torch.reshape(pts, [-1, 3])
                dir_samples = torch.reshape(dir_samples, [-1, 3])

                optic_d = self._run_network(pts, dir_samples, self.model_fine)
                optic_d = torch.reshape(optic_d, [M, N + self.config.N_importance, 4])

                rgb_map_fine, _ = self._composite(optic_d, z_vals, ray_dirs)
            else:
                rgb_map_fine = rgb_map_coarse

            res_ls["rgb_map_coarse"].append(rgb_map_coarse)
            res_ls["rgb_map_fine"].append(rgb_map_fine)

        res = {k: torch.cat(res_ls[k], 0) for k in res_ls}

        return res["rgb_map_fine"], res["rgb_map_coarse"]

    def _run_network(self, pts, view_dirs, model):
        inputs_flat = pts
        embedded = self.embedder_p(inputs_flat)

        if view_dirs is not None:
            input_dirs_flat = view_dirs
            embedded_dirs = self.embedder_d(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        # outputs_flat = batchify(fn, netchunk)(embedded)
        chunk = self.config.netchunk
        outputs_flat = torch.cat(
            [
                model(embedded[i : i + chunk])
                for i in range(0, embedded.shape[0], chunk)
            ],
            0,
        )
        # outputs = torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs_flat

    def _transfer(self, optic_d, dists):
        rgbs = torch.sigmoid(optic_d[..., :3])  # (chunk_size, N_samples, 3)
        alphas = 1.0 - torch.exp(
            -F.relu(optic_d[..., 3]) * dists
        )  # (chunk_size, N_samples)

        return rgbs, alphas

    def _composite(self, optic_d, s_samples, rays_d):
        # distances between each samples
        dists = s_samples[..., 1:] - s_samples[..., :-1]  # (chunk_size, N_samples - 1)
        dists_list = [dists, torch.tensor([1e10]).expand(dists[..., :1].shape)]
        dists = torch.cat(dists_list, -1)  # (chunk_size, N_samples)

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # retrieve display colors and alphas for each samples by a transfer function
        rgbs, alphas = self._transfer(optic_d, dists)

        weights = alphas * torch.cumprod(
            torch.cat([torch.ones((alphas.shape[0], 1)), 1.0 - alphas + 1e-10], dim=-1)[
                :, :-1
            ],
            dim=-1,
        )  # (chunk_size, N_samples)
        rgb_map = torch.sum(weights[..., None] * rgbs, dim=-2)  # (chunk_size, 3)
        acc_map = torch.sum(weights, -1)  # (chunk_size)

        if self.config.white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        return rgb_map, weights
