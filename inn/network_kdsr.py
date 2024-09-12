import math
import torch
import torch.nn as nn
import torch.nn.functional as F
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.LeakyReLU(0.1, inplace=True), res_scale=1):

        super(ResBlock, self).__init__()

        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        # self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, act=False, bias=True):
        m = []
        if (int(scale) & (int(scale) - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class IDR_DDC(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(IDR_DDC, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(channels_in, channels_in, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(channels_in, channels_in * self.kernel_size * self.kernel_size, bias=False)
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        # branch 1
        kernel = self.kernel(x[1]).view(-1, 1, self.kernel_size, self.kernel_size)
        out = F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2)
        out = out.view(b, -1, h, w)


        return out



class IDR_DCRB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(IDR_DCRB, self).__init__()

        self.da_conv1 = IDR_DDC(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.relu =  nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''

        out = self.relu(self.da_conv1(x))
        out = self.conv1(out)
        out = out  + x[0]

        return out


class DAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(DAG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            IDR_DCRB(conv, n_feat, kernel_size, reduction) \
            for _ in range(n_blocks)
        ]
        # modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        res = x[0]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1]])

        return res


class KDSR(nn.Module):
    def __init__(self, n_feats=128, scale=4,n_sr_blocks = 42, rgb_range=1,reduction = 8, conv=default_conv):
        super(KDSR, self).__init__()
        kernel_size = 3

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)


        # body
        modules_body = [
            DAG(default_conv, n_feats, kernel_size, reduction, n_sr_blocks)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, k_v):

        # sub mean
        x = self.sub_mean(x)

        # head
        x = self.head(x)

        # body
        res = x
        res = self.body[0]([res, k_v])
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        # add mean
        x = self.add_mean(x)

        return x



class KD_IDE(nn.Module):
    def __init__(self,n_feats = 128, n_encoder_res = 6):
        super(KD_IDE, self).__init__()
        E1=[nn.Conv2d(3, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        E2=[
            ResBlock(
                default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3=[
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E=E1+E2+E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )
        self.compress = nn.Sequential(
            nn.Linear(n_feats*4, n_feats),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        S_fea = []
        fea1 = self.mlp(fea)
        fea = self.compress(fea1)
        S_fea.append(fea1)
        return fea,S_fea


class BlindSR_ST(nn.Module):
    def __init__(self, n_feats=128, n_encoder_res=6, scale=4,n_sr_blocks=42 ):
        super(BlindSR_ST, self).__init__()

        # Generator
        self.G = KDSR(n_feats=n_feats, scale=scale,n_sr_blocks=n_sr_blocks)

        self.E_st = KD_IDE(n_feats=n_feats, n_encoder_res=n_encoder_res)

        self.pixel_unshuffle = nn.PixelUnshuffle(scale)




    def forward(self, x):
        if self.training:
            # degradation-aware represenetion learning
            deg_repre, S_fea = self.E_st(x)

            # degradation-aware SR
            sr = self.G(x, deg_repre)

            return sr, S_fea
        else:
            # degradation-aware represenetion learning
            deg_repre, _ = self.E_st(x)

            # degradation-aware SR
            sr = self.G(x, deg_repre)


            return sr,deg_repre

