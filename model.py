import torch.nn as nn
import numpy as np

from nn import *

q_size = 16

class PixelSNAIL(nn.Module):

    def __init__(self, obs_dim, dropout_p=0.5, nr_resnet=5,
                 nr_filters=256, attn_rep=12, nr_logistic_mix=10):
        super().__init__()
        self.attn_rep = attn_rep
        self.nr_resnet = nr_resnet

        self.conv1 = down_shifted_conv2d(obs_dim[0] + 1, nr_filters,
                                         filter_size=(1, 3))
        self.conv2 = down_right_shifted_conv2d(obs_dim[0] + 1, nr_filters,
                                               filter_size=(2, 1))
        self.attention_blocks = nn.ModuleList()
        for attn_rep in range(attn_rep):
            attn_block = nn.ModuleList()
            for rep in range(nr_resnet):
                attn_block.append(gated_resnet(nr_filters, down_right_shifted_conv2d))
            attn_block.append(gated_resnet(5 + nr_filters, nin)) # rn
            attn_block.append(nin(5 + nr_filters, nr_filters // 2 + q_size)) # rn+1
            attn_block.append(gated_resnet(2 + nr_filters, nin)) # rn+2
            attn_block.append(nin(2 + nr_filters, q_size)) # rn+3
            attn_block.append(causal_attention(obs_dim)) # rn+4
            attn_block.append(gated_resnet(nr_filters, nin)) # rn+5
            self.attention_blocks.append(attn_block)
        self.nin_out = nin(nr_filters, 10 * nr_logistic_mix)

        self.register_buffer('background', self._create_background(obs_dim))

    def _create_background(self, obs_dim):
        _, h, w = obs_dim
        base = np.zeros((h, w)).astype(np.float32)
        h_pos = np.arange(h)[None, None, :, None] + base
        w_pos = np.arange(w)[None, None, None, :] + base
        background = np.concatenate((h_pos, w_pos), axis=1).astype(np.float32)
        return torch.FloatTensor(background)

    def forward(self, x):
        xs = x.shape
        x_pad = torch.cat((x, torch.ones_like(x[:, [0], :, :])), dim=1)

        ul_list = [down_shift(self.conv1(x_pad)) + right_shift(self.conv2(x_pad))]

        for attn_rep in self.attn_rep:
            attn_block = self.attention_blocks[attn_rep]
            for rep in range(self.nr_resnet):
                ul_list.append(attn_block[rep](ul_list[-1]))

            rn = self.nr_resnet
            ul = ul_list[-1]
            raw_content = torch.cat((x, ul, self.background), dim=1)
            raw = attn_block[rn+1](attn_block[rn](raw_content))
            key, mixin = raw[:, :q_size, :, :], raw[:, q_size:, :, :]
            raw_q = torch.cat((ul, self.background), dim=1)
            query = attn_block[rn+3](attn_block[rn+2](raw_q))
            mixed = attn_block[rn+4](key, mixin, query)

            ul_list.append(attn_block[rn+5](ul, mixed))
        out = self.nin_out(ul_list[-1])
        return out
