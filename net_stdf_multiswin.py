import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.dcn.deform_conv import ModulatedDeformConv
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import swin_transformer as st


class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class invPixelShuffle(nn.Module):

    def __init__(self, ratio=2):
        super(invPixelShuffle, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        ratio = self.ratio
        b, ch, y, x = tensor.size()

        return tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).permute(0, 1, 3, 5, 2, 4).contiguous().view(b,
                                                                                                                    -1,
                                                                                                                    y // ratio,
                                                                                                                    x // ratio)


class QFAttention(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC'):
        super(QFAttention, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'

        self.res = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        res = (gamma) * self.res(x) + beta
        return x + res


# ==========
# Spatio-temporal deformable fusion module
# ==========

class STDF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True)
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )

        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2
        )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks // 2, deformable_groups=in_nc
        )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
            )

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        off_msk = self.offset_mask(self.out_conv(out))
        off = off_msk[:, :in_nc * 2 * n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc * 2 * n_off_msk:, ...]
        )

        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk),
            inplace=True
        )

        return fused_feat


# ==========
# Quality enhancement module
# ==========


class PlainCNN(nn.Module):
    def __init__(self, in_nc=64, nf=48, nb=8, out_nc=3, base_ks=3):
        """
        Args:
            in_nc: num of input channels from STDF.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super(PlainCNN, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=1),
            nn.ReLU(inplace=True)
        )

        hid_conv_lst = []
        for _ in range(nb - 3):
            hid_conv_lst += [
                nn.Conv2d(nf, nf, base_ks, padding=1),
                nn.ReLU(inplace=True)
            ]
        self.hid_conv = nn.Sequential(*hid_conv_lst)

        self.out_conv = nn.Conv2d(nf, out_nc, base_ks, padding=1)

    def forward(self, inputs):
        out = self.in_conv(inputs)
        out = self.hid_conv(out)
        out = self.out_conv(out)
        return out


class HDRO(nn.Module):
    """
    Hybrid Dilation Reconstruction Operator
    """
    def __init__(self, nf=64, out_nc=1, base_ks=3, bias=True):
        super(HDRO, self).__init__()

        self.dilation_1 = nn.Sequential(nn.Conv2d(nf, out_nc, 3, stride=1, padding=1, dilation=1, bias=True),
                                        )
        self.dilation_2 = nn.Sequential(nn.Conv2d(nf, out_nc, 3, stride=1, padding=2, dilation=2, bias=True),
                                        )
        self.dilation_3 = nn.Sequential(nn.Conv2d(nf, out_nc, 3, stride=1, padding=4, dilation=4, bias=True),
                                        )
        self.scale = nn.Parameter(torch.zeros((1, nf, 1, 1)), requires_grad=True)
        # self.dwconv = nn.Sequential(
        #     nn.Conv2d(nf*3, nf*3, 1),
        #     nn.Conv2d(nf*3, nf*3, 3, 1, 1, groups=nf*3),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(nf*3, nf, 1)
        # )
        # self.conv1 = nn.Conv2d(nf, nf, 1, 1, 0)
        self.pwconv = nn.Sequential(nn.Conv2d(nf*3, nf, 1, 1, 0),
                                  nn.ReLU(inplace=True))
        # self.conv = nn.Conv2d(out_nc*3, out_nc, base_ks, padding=(base_ks//2),stride = 1, bias=bias)
        self.conv = nn.Sequential(nn.Conv2d(out_nc, out_nc, 3, 1, 1),
                                  nn.ReLU(inplace=True))
    def forward(self, fea):
        fea1 = self.dilation_1(fea)
        fea2 = self.dilation_2(fea)
        fea3 = self.dilation_3(fea)
        # out_fea = self.dwconv(torch.cat([fea1,fea2,fea3],dim=1)) * self.conv1(fea)
        out_fea = self.conv(self.pwconv(torch.cat([fea1, fea2, fea3], dim=1)))

        return out_fea


class Select_HDRO(nn.Module):
    """
    Hybrid Dilation Reconstruction Operator
    """

    def __init__(self, in_nc=64, nf=64, out_nc=1, base_ks=3, bias=True):
        super(Select_HDRO, self).__init__()

        # self.inconv = nn.Sequential(nn.Conv2d(in_nc, nf, 3, 1, 1),
        #                           nn.ReLU(inplace=True))

        self.dilation_1 = nn.Sequential(nn.Conv2d(nf, out_nc, 3, stride=1, padding=1, dilation=1, bias=True),
                                        )
        self.dilation_2 = nn.Sequential(nn.Conv2d(nf, out_nc, 3, stride=1, padding=2, dilation=2, bias=True),
                                        )
        self.dilation_3 = nn.Sequential(nn.Conv2d(nf, out_nc, 3, stride=1, padding=4, dilation=4, bias=True),
                                        )
        # self.scale = nn.Parameter(torch.zeros((1, out_nc, 1, 1)), requires_grad=True)

        self.skff = SKFF(in_channels=out_nc)
        # self.dwconv = nn.Sequential(
        #     nn.Conv2d(nf*3, nf*3, 1),
        #     nn.Conv2d(nf*3, nf*3, 3, 1, 1, groups=nf*3),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(nf*3, nf, 1)
        # )
        # self.conv1 = nn.Conv2d(nf, nf, 1, 1, 0)
        # self.pwconv = nn.Sequential(nn.Conv2d(nf * 3, nf, 1, 1, 0),
        #                             nn.ReLU(inplace=True))
        # self.conv = nn.Conv2d(out_nc*3, out_nc, base_ks, padding=(base_ks//2),stride = 1, bias=bias)
        self.conv = nn.Sequential(nn.Conv2d(out_nc, out_nc, 3, 1, 1),
                                  nn.ReLU(inplace=True))

    def forward(self, fea):

        # fea = self.inconv(fea)
        fea1 = self.dilation_1(fea)
        fea2 = self.dilation_2(fea)
        fea3 = self.dilation_3(fea)
        # out_fea = self.dwconv(torch.cat([fea1,fea2,fea3],dim=1)) * self.conv1(fea)
        out_fea = self.conv(self.skff([fea1, fea2, fea3]))

        return out_fea

class ParallelCNN(nn.Module):
    def __init__(self, in_nc=64, nf=48, nb=8, out_nc=3, base_ks=3, num_blocks=5):
        """
        Args:
            in_nc: num of input channels from STDF.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super(ParallelCNN, self).__init__()

        # self.in_conv = nn.Sequential(
        #     nn.Conv2d(in_nc, nf, base_ks, padding=1),
        #     nn.ReLU(inplace=True)
        # )

        self.out_conv = nn.Sequential(
            # nn.Conv2d(in_nc, nf, base_ks, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(nf, 1, base_ks, padding=1)
        )

        self.layers = nn.ModuleList()
        i = 0
        for i_layer in range(num_blocks):
            # layer = HDRO(nf=nf,
            #              out_nc=nf)
            if i == 0:
                layer = Select_HDRO(nf=in_nc,
                                out_nc=nf)
                self.layers.append(layer)
            else:
                layer = Select_HDRO(nf=nf,
                                    out_nc=nf)
                self.layers.append(layer)
            i += 1

    def forward(self, x):

        # x = self.in_conv(x)

        for layer in self.layers:
            x = layer(x)

        x = self.out_conv(x)

        return x


class Mid_layer(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, embed_dim=96, depths=[2, 2, 2, 2],
                 num_heads=[3, 6, 12, 24], window_size=[4, 3], mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="Dual up-sample", **kwargs):
        super(Mid_layer, self).__init__()

        num_layers = len(depths)
        # self.embed_dim = embed_dim
        self.ape = ape
        #self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        #self.num_features_up = int(embed_dim * 2)
        #self.prelu = nn.PReLU()
        # self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.invshuffle = invPixelShuffle()
        self.in_conv = nn.Conv2d(in_chans * 4, in_chans, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_chans, in_chans * 4, kernel_size=3, stride=1, padding=1)
        self.upshuffle = nn.PixelShuffle(2)

        self.patch_embed = st.PatchEmbed_noConv(in_chans=embed_dim, embed_dim=embed_dim,
                                                norm_layer=norm_layer if patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.alpha = []
        self.beta = []
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        for _ in range(num_layers):
            self.alpha.append(torch.nn.Parameter(torch.Tensor([1.0]), requires_grad=True).to(device))
            self.beta.append(torch.nn.Parameter(torch.Tensor([1.0]), requires_grad=True).to(device))

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.swin_convs = nn.ModuleList()
        for i in range(num_layers):
            swin_conv = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.swin_convs.append(swin_conv)

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(num_layers):
            # print(dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])])
            layer = st.BasicLayer_multishift(  # dim=int(embed_dim * 2 ** i_layer),
                dim=int(embed_dim),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                # downsample=st.PatchMerging_noconv if (i_layer < self.num_layers - 1) else None,
                downsample=None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

            # if final_upsample == "Dual up-sample":
            #     # self.up = UpSample(in_channels=embed_dim, scale_factor=4)
            #     # self.output = nn.Conv2d(in_channels=embed_dim // 2, out_channels=self.out_chans, kernel_size=3,
            #     #                         stride=1,
            #     #                         padding=1, bias=False)  # kernel = 1
            #     self.output = nn.Conv2d(in_channels=embed_dim, out_channels=out_chans * 4, kernel_size=3,
            #                             stride=1,
            #                             padding=1, bias=False)  # kernel = 1

            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def up_x4(self, x, hw_shape):
        H, W = hw_shape
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"
        #
        x = x.view(B, H, W, C)  # B, H, W, C
        x = x.permute(0, 3, 1, 2)  # B,C,H,W

        return x

    def forward(self, x):
        # out = self.conv_first(x)
        out = self.invshuffle(x)
        out = self.in_conv(out)
        # Encoder and Bottleneck
        # x, residual, x_downsample = self.forward_features(x)
        out, hw_shape = self.patch_embed(out)
        if self.ape:
            out = out + self.absolute_pos_embed
        out = self.pos_drop(out)
        # x_downsample = []
        i = 0
        for layer, swin_conv in zip(self.layers, self.swin_convs):
        # for layer in self.layers:
            # out1 = self.up_x4(out, hw_shape)
            # out1 = swin_conv(self.up_x4(out, hw_shape))
            # out1, hw_shape = self.patch_embed(swin_conv(self.up_x4(out, hw_shape)))
            # # x_downsample.append(x)
            # out1, hw_shape = layer(self.patch_embed(swin_conv(self.up_x4(out, hw_shape))), hw_shape)
            out = layer(self.patch_embed(swin_conv(self.up_x4(out, hw_shape)))[0], hw_shape)[0] * self.beta[i] + out * \
                  self.alpha[i]
            # out1, hw_shape = layer(out, hw_shape)
            # out = out1 * self.beta[i]
            i = i + 1

        out = self.up_x4(out, hw_shape)
        out = self.out_conv(out)
        out = x + self.upshuffle(out)

        # out = self.output(out)

        return out


# ==========
# MFVQE network
# ==========

class MFVQE(nn.Module):
    """STDF -> QE -> residual.

    in: (B T C H W)
    out: (B C H W)
    """

    def __init__(self, opts_dict):
        """
        Arg:
            opts_dict: network parameters defined in YAML.
        """
        super(MFVQE, self).__init__()

        self.radius = opts_dict['network']['radius']
        self.input_len = 2 * self.radius + 1
        self.in_nc = opts_dict['network']['stdf']['in_nc']
        self.ffnet = STDF(
            in_nc=self.in_nc * self.input_len,
            out_nc=opts_dict['network']['stdf']['out_nc'],
            nf=opts_dict['network']['stdf']['nf'],
            nb=opts_dict['network']['stdf']['nb'],
            deform_ks=opts_dict['network']['stdf']['deform_ks']
        )
        self.midnet = Mid_layer(
            patch_size=opts_dict['network']['swin']['patch_size'],
            in_chans=opts_dict['network']['swin']['in_chans'],
            out_chans=opts_dict['network']['swin']['out_chans'],
            embed_dim=opts_dict['network']['swin']['embed_dim'],
            depths=opts_dict['network']['swin']['depths'],
            num_heads=opts_dict['network']['swin']['num_heads'],
            window_size=opts_dict['network']['swin']['window_size'],
            mlp_ratio=opts_dict['network']['swin']['mlp_ratio'],
            qkv_bias=opts_dict['network']['swin']['qkv_bias'],
            qk_scale=opts_dict['network']['swin']['qk_scale'],
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            final_upsample="Dual up-sample"
        )
        # self.qenet = PlainCNN(
        #     in_nc=opts_dict['network']['qenet']['in_nc'],
        #     nf=opts_dict['network']['qenet']['nf'],
        #     nb=opts_dict['network']['qenet']['nb'],
        #     out_nc=opts_dict['network']['qenet']['out_nc']
        # )
        # self.qenet = trans_restormer(
        #     dim=opts_dict['network']['qenet']['nf'],
        #     num_heads=opts_dict['network']['qenet']['num_heads'],
        #     num_blocks=opts_dict['network']['qenet']['num_blocks']
        # )
        self.qenet = ParallelCNN(
            in_nc=opts_dict['network']['qenet']['in_nc'],
            nf=opts_dict['network']['qenet']['nf'],
            nb=opts_dict['network']['qenet']['nb'],
            out_nc=opts_dict['network']['qenet']['out_nc'],
            num_blocks=opts_dict['network']['qenet']['num_blocks']
        )
        # self.qenet = AttQE(
        #     nf=opts_dict['network']['qenet']['nf'],
        #     num_layer=opts_dict['network']['qenet']['num_block'],
        #     block_layer=opts_dict['network']['qenet']['block_layer'],
        # )

        # self.out_conv = nn.Conv2d(opts_dict['network']['stdf']['out_nc'], 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.ffnet(x)
        out = self.midnet(out)
        # out = self.out_conv(out)
        out = self.qenet(out)
        # out = self.upshuffle(out)
        # e.g., B C=[B1 B2 B3 R1 R2 R3 G1 G2 G3] H W, B C=[Y1 Y2 Y3] H W or B C=[B1 ... B7 R1 ... R7 G1 ... G7] H W
        frm_lst = [self.radius + idx_c * self.input_len for idx_c in range(self.in_nc)]
        out += x[:, frm_lst, ...]  # res: add middle frame          F
        return out


import yaml
import argparse
import os
from thop import profile

if __name__ == "__main__":
    def receive_arg():
        """Process all hyper-parameters and experiment settings.

        Record in opts_dict."""

        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--opt_path', type=str, default='config/option_R3_mfqev2_1G.yml',
            help='Path to option YAML file.'
        )
        parser.add_argument(
            '--local_rank', type=int, default=0,
            help='Distributed launcher requires.'
        )
        args = parser.parse_args()

        with open(args.opt_path, 'r') as fp:
            opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

        return opts_dict


    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opts_dict = receive_arg()
    x = torch.randn(1, 7, 720, 720).cuda()  # .cuda()#.to(torch.device('cuda'))
    model = MFVQE(opts_dict=opts_dict)
    model = model.to(0)
    # print(model)
    # y = model(x)

    # print(y.shape)
    print('-' * 50)
    print('#generator parameters:', sum(param.numel() for param in model.parameters()))
    print(profile(model, inputs=(x, )))
