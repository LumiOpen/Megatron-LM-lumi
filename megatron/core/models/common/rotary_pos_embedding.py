# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import importlib.util
import math 
import torch
from torch import einsum, nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta,rope_scaling: bool = False, seq_len_interpolation_factor=None):
        super().__init__()
        self.theta = theta
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2).float() / dim))
        if rope_scaling:
            self.inv_freq = self._apply_scaling(self.inv_freq)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, max_seq_len, offset=0):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        if self.seq_len_interpolation_factor is not None:
            seq = seq.type_as(self.inv_freq)
            seq *= 1 / self.seq_len_interpolation_factor
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        return emb[:, None, None, :]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(f'{prefix}inv_freq', None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
    
    def _apply_scaling(
        self,
        freqs,
        factor=8,
        low_freq_factor=1,
        high_freq_factor=4,
        original_max_position_embeddings=8192,
    ):
        # This implementation is adapted from:
        # https://github.com/huggingface/transformers/blob/2a5a6ad18aa22e98429bb5ecb880660328030ea0/src/transformers/modeling_rope_utils.py#L303-L343

        factor = factor  # `8` in the original implementation
        low_freq_factor = low_freq_factor  # `1` in the original implementation
        high_freq_factor = high_freq_factor  # `4` in the original implementation
        old_context_len = original_max_position_embeddings  # `8192` in the original implementation

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / freqs
        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, freqs / factor, freqs)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            1 - smooth_factor
        ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        return inv_freq_llama

def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """

    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1).bfloat16()
