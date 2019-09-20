import common
import torch
import torch.nn as nn
import numpy as np


class resLF(nn.Module):
    def __init__(self, conv=common.default_conv, n_view=9, scale=2):
        super(resLF, self).__init__()

        # 4 resblock in each image stack
        n_resblock = 4
        # 4 resblock in the global part
        n_mid_resblock = 4
        n_feats = 32
        kernel_size = 3
        act = nn.ReLU(True)
        n_colors = 1

        self.n_view = n_view

        # define head module
        m_head = [conv(n_view, n_feats, kernel_size)]
        central_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_mid_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(n_mid_resblock)
        ]

        m_body = [
            common.ResBlock(
                conv, n_feats * 4, kernel_size, act=act, res_scale=1
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats * 4, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, n_colors, kernel_size,
                padding=(kernel_size // 2)
            )
        ]

        self.head = nn.Sequential(*m_head)
        self.central_head = nn.Sequential(*central_head)
        self.midbody = nn.Sequential(*m_mid_body)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, train_data_0, train_data_90, train_data_45, train_data_135):

        # extract the central view from the image stack
        mid_view = np.int8((self.n_view - 1) / 2)
        central_x = train_data_0[:, mid_view:mid_view + 1, :, :]

        res_x = self.central_head(central_x)

        y = self.head(train_data_0)
        mid_0d = self.midbody(y)
        y = self.head(train_data_90)
        mid_90d = self.midbody(y)
        y = self.head(train_data_45)
        mid_45d = self.midbody(y)
        y = self.head(train_data_135)
        mid_135d = self.midbody(y)

        ''' Merge layers '''
        mid_merged = torch.cat((mid_0d, mid_90d, mid_45d, mid_135d), 1)
        res = self.body(mid_merged)

        res += res_x

        x = self.tail(res)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
