import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Embedder(nn.Module):
    def __init__(self, input_dims, max_freq_pow, num_freqs, periodic_fns=[torch.sin, torch.cos],
                 log_sampling=True, include_input=True):

        super(Embedder, self).__init__()
        
        embed_fns = []
        out_dims = 0
        if include_input:
            embed_fns.append(lambda x : x)
            out_dims += input_dims
        
        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0., max_freq_pow, steps=num_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq_pow, steps=num_freqs)
            
        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dims += input_dims
                
        self.embed_fns = embed_fns
        self.out_dims = out_dims
        
    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)
