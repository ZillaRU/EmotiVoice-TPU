import math
import torch
from torch import nn
from torch.nn import functional as F
import time
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from .commons import init_weights, get_padding, sequence_mask
from . import modules
from models import EngineOV
import numpy as np



class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.model_core = EngineOV('./model_file/converter/myenc_f32.bmodel', device_id=0)
        self.out_channels = out_channels


    def forward(self, x, x_lengths, g=None, tau=1.0, max_len=1024):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, max_len), 1).to(x.dtype)
        
        if x.shape[-1] < max_len: # padding
            x = torch.cat([x, torch.zeros([*(x.shape[:-1]), max_len-x.shape[-1]])], axis=2)
        elif x.shape[-1] > max_len:
            x = x[..., :max_len]
            print(f'===== WARNING =====: Your input ({x.shape[-1]}) exceeds the length limit ({max_len}). The output is incomplete.')
        stats = self.model_core([x.numpy().astype(np.float32), x_mask.numpy().astype(np.float32), g.numpy().astype(np.float32)])[0]
        m, logs = torch.split(torch.from_numpy(stats), self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * tau * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=256,
        gin_channels=256,
        **kwargs
    ):
        super().__init__()

        self.dec = EngineOV('./model_file/converter/decoder_2048_f16.bmodel', device_id=0)

        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )

        self.flow = EngineOV('./model_file/converter/flow_f32.bmodel', device_id=0)

        self.flow_reverse = EngineOV('./model_file/converter/flow_reverse_f32.bmodel', device_id=0)

        self.ref_enc = EngineOV('./model_file/converter/ref_enc_F32.bmodel', device_id=0)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt, tau=1.0, max_len=2048):
        g_src = sid_src
        g_tgt = sid_tgt
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src, tau=tau)
        z_p = self.flow([z.numpy().astype(np.float32), y_mask.numpy().astype(np.float32), g_src.numpy().astype(np.float32)])[0]
        z_hat = self.flow_reverse([z_p, y_mask.numpy().astype(np.float32), g_tgt.numpy().astype(np.float32)])[0]
        z_p = torch.from_numpy(z_p)
        z_hat = torch.from_numpy(z_hat)
        if max_len > z_hat.shape[2]:
            temp = torch.cat([z_hat*y_mask, torch.zeros(z_hat.shape[0],z_hat.shape[1],max_len-z_hat.shape[2])], axis=2)
        else:
            temp = (z_hat*y_mask)[z_hat.shape[0],z_hat.shape[1], :max_len]
            print(f'===== WARNING =====: Your input ({y_lengths}) exceeds the length limit ({max_len}). The output is incomplete.')
        o_hat = torch.from_numpy(self.dec([temp.numpy(), g_tgt.numpy()])[0])[:,:,:(y_lengths*256)]
        return o_hat, y_mask, (z, z_p, z_hat)
