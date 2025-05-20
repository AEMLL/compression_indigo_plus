import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

def details_to_tensor(x):
    h, w = max(d.shape[2] for d in x), max(d.shape[3] for d in x)
    lu = []
    for t in x:
        t = pad_to_target(t, w, h)
        lu.append(t)
    return torch.cat(lu, dim=1)

def tensor_to_details(x):
    lp = []
    num_channels = x.shape[1] // 3
    for ch in range(3):
        t = x[:, ch: ch + num_channels, :, :]
        lp.append(t)
    return lp

def pad_to_target(t, w, h):
    pd = (0, w - t.shape[3], 0, h - t.shape[2], 0, 0, 0, 0)
    return F.pad(input=t, pad=pd)

class AdaptiveReLU(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.threshold = torch.nn.Parameter(-0.1 * torch.rand(1, in_dim, 1, 1))

    def forward(self, x, aug):
        return torch.relu(x + self.threshold * aug)

class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, p, bias=False, padding='same'):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(in_dim, in_dim, kernel_size=p, padding=padding, groups=in_dim, bias=False)
        self.pointwise = torch.nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class ResidualBlock_CRes_kdsr(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(latent_dim, latent_dim, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(latent_dim, latent_dim, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
        )

    def forward(self, x, sigma):
        local_scale = self.fc(sigma)
        out = self.conv1(x)
        out = self.conv2(self.act(out))
        return x + out * local_scale.view(-1, 32, 1, 1)

class ResNet(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim, p, iterations, nltyp, convtyp, basic_block_type):
        super().__init__()
        self.convin = nn.Conv2d(in_dim, latent_dim, p, padding='same', bias=False)

        if basic_block_type == "rb_cres_kdsr":
            self.blocks = nn.ModuleList(ResidualBlock_CRes_kdsr(latent_dim) for _ in range(iterations))

        self.convout = nn.Conv2d(latent_dim, out_dim, p, padding='same', bias=False)

    def forward(self, x, cond):
        x = self.convin(x)
        for block in self.blocks:
            x = block(x, cond)
        return self.convout(x)

class PredictorNet(nn.Module):
    def __init__(self, num_channels, latent_dim, p, iterations, nltyp, convtyp, basic_block_type):
        super().__init__()
        self.net = ResNet(num_channels, latent_dim, 3 * num_channels, p, iterations, nltyp, convtyp, basic_block_type)

    def forward(self, x, cond):
        x = self.net(x, cond)
        return tensor_to_details(x)

class UpdaterNet(nn.Module):
    def __init__(self, num_channels, latent_dim, p, iterations, nltyp, convtyp, basic_block_type):
        super().__init__()
        self.net = ResNet(3 * num_channels, latent_dim, num_channels, p, iterations, nltyp, convtyp, basic_block_type)

    def forward(self, x, cond):
        x = details_to_tensor(x)
        return self.net(x, cond)

class LiftNet(nn.Module):
    def __init__(self, num_channels, LiftNet_iter, basic_block_type, nltyp_adarelu=1):
        super().__init__()
        latent_dim = 32
        p = 3
        iterations = LiftNet_iter

        if nltyp_adarelu:
            nltyp = AdaptiveReLU
        else:
            nltyp = SoftThreshold

        convtyp = DepthwiseSeparableConvolution
        self.p = PredictorNet(num_channels, latent_dim, p, iterations, nltyp, convtyp, basic_block_type)
        self.u = UpdaterNet(num_channels, latent_dim, p, iterations, nltyp, convtyp, basic_block_type)

    def forward(self, x, cond):
        c, d = x
        pd = self.p(c, cond)
        nd = []
        for i in range(len(d)):
            nd.append(d[i] + pd[i][..., :d[i].shape[-2], :d[i].shape[-1]])
        c = c + self.u(nd, cond)
        return c, nd

    def inverse(self, x, cond):
        c, d = x
        c = c - self.u(d, cond)
        pd = self.p(c, cond)
        nd = []
        for i in range(len(d)):
            nd.append(d[i] - pd[i][..., :d[i].shape[-2], :d[i].shape[-1]])
        return c, nd

class LazyWaveletTransform(nn.Module):
    def __init__(self, undecimated):
        super().__init__()
        self.undecimated = undecimated

    def forward(self, x):
        ec, oc = self._1d_forward(x, dim=3)
        ecer, ecor = self._1d_forward(ec, dim=2)
        ocer, ocor = self._1d_forward(oc, dim=2)
        return ecer, [ecor, ocer, ocor]

    def _1d_forward(self, x, dim):
        e = x[..., ::2] if dim == 3 else x[..., ::2, :]
        o = x[..., 1::2] if dim == 3 else x[..., 1::2, :]
        return e, o

    def inverse(self, x):
        ecer, [ecor, ocer, ocor] = x
        oc = self._1d_inverse((ocer, ocor), dim=2)
        ec = self._1d_inverse((ecer, ecor), dim=2)
        return self._1d_inverse((ec, oc), dim=3)

    def _1d_inverse(self, x, dim):
        e, o = x
        sz = (e.shape[0], e.shape[1], e.shape[2], e.shape[3] + o.shape[3]) if dim == 3 \
            else (e.shape[0], e.shape[1], e.shape[2] + o.shape[2], e.shape[3])
        ix = torch.zeros(sz, device=e.device)
        if dim == 3:
            ix[..., ::2] = e
            ix[..., 1::2] = o
        else:
            ix[..., ::2, :] = e
            ix[..., 1::2, :] = o
        return ix

class InvertibleNet(nn.Module):
    def __init__(self, num_lifts, num_channels, LiftNet_iter, basic_block_type, nltyp_adarelu=1):
        super().__init__()
        self.lifts = nn.ModuleList(LiftNet(num_channels=num_channels, LiftNet_iter=LiftNet_iter, nltyp_adarelu=nltyp_adarelu, basic_block_type=basic_block_type) for _ in range(num_lifts))

    def forward(self, x, cond):
        for lift in self.lifts:
            x = lift(x, cond)
        return x

    def inverse(self, x, cond):
        for lift in reversed(self.lifts):
            x = lift.inverse(x, cond)
        return x

class ImageTransform(nn.Module):
    def __init__(self, num_channels, LiftNet_iter=7, offset=0.5, nltyp_adarelu=1, basic_block_type="rb_cres_kdsr"):
        super().__init__()
        self.num_channels = num_channels
        self.num_lifts = 4
        self.split = LazyWaveletTransform(undecimated=False)
        self.inn = InvertibleNet(num_lifts=self.num_lifts, num_channels=self.num_channels, LiftNet_iter=LiftNet_iter, nltyp_adarelu=nltyp_adarelu, basic_block_type=basic_block_type)
        self.offset = offset

    def forward(self, x, cond, J=float('inf')):
        x = x - self.offset
        d = deque()
        while x.shape[2] > 7 and x.shape[3] > 7 and J > 0:
            x = self.split(x)
            x, yd = self.inn(x, cond)
            d.appendleft(yd)
            J -= 1
        x = x + self.offset
        return [x, d], cond

    def inverse_from_pieces(self, x, cond):
        yc = x[0] - self.offset
        for k in range(len(x[1])):
            yc = self.inn.inverse((yc, x[1][k]), cond)
            yc = self.split.inverse(yc)
        return yc + self.offset

class RWNN(nn.Module):
    def __init__(self, LiftNet_iter=7, offset=0.5, nltyp_adarelu=1, basic_block_type="rb_cres_kdsr", net_kdsr=None, kdsr=1):
        super().__init__()
        self.transform_net = ImageTransform(num_channels=3, LiftNet_iter=LiftNet_iter, offset=offset, nltyp_adarelu=nltyp_adarelu, basic_block_type=basic_block_type)
        self.net_kdsr = net_kdsr
        self.kdsr = kdsr

    def dae_f(self, x, cond=None, layer=2):
        [cn, dn], _ = self.transform_net.forward(x, cond, J=layer)
        return cn, dn

    def dae_b(self, cn, dn, cond=None):
        res = self.transform_net.inverse_from_pieces([cn, dn], cond)
        return res
