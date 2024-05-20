import torch   
import os
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch.nn.functional as F


from torch.nn import functional as F
import numpy as np
import random

seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
print('random seed',seed)
whole_idx=0

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
class LayerNormFunction(torch.autograd.Function):
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)


class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

import math
class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) 
class attention2d(nn.Module):
    def __init__(self, c_in, c_out,):
        super(attention2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c_in, c_out, 1,)
        self.fc2 = nn.Conv2d(c_out, c_out, 1,)
        self.act = GeLU()
    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x).view(x.size(0),1,-1,1)
        return F.softmax(x, -2) 
      
class basis_based_conv(nn.Module):
    def __init__(self,  in_channels = 16,
                        out_channels = 16,
                        kernel_size = 1,
                        embedding_dim = 16,
                        padding=0, 
                        stride=1, 
                        bias =False):
        super(basis_based_conv, self).__init__()
        
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.embedding_dim=embedding_dim
        
        self.padding=padding
        self.stride=stride
        
        self.train_embedding = torch.nn.Embedding(in_channels*out_channels, embedding_dim)
        self.train_img_k = attention2d(in_channels, embedding_dim)
        # self.train_mlp = nn.Linear(embedding_dim,1)
        
        self.train_mlp = nn.Linear(embedding_dim,int(embedding_dim/2))
        self.gelu=GeLU()
        self.train_mlp2 = nn.Linear(int(embedding_dim/2),1)

    def forward(self, x, model=None):
        
        ########################Initialization
        b,c,h,w = x.shape
        index = torch.arange(0, self.in_channels*self.out_channels,device=x.device)
        index = index.unsqueeze(0).expand(b, -1)
        param = self.train_embedding(index)

        ########################Normalization
        norms_param = torch.norm(param, dim=-1, keepdim=True)
        param = param / norms_param
        param_reshaped = param.view(b, -1, self.embedding_dim, 1)
        param = torch.matmul(param_reshaped, param_reshaped.transpose(2, 3))
        
        ########################Orthogonalization
        I = torch.zeros_like(param)
        indices = torch.arange(self.embedding_dim)
        I[:, :, indices, indices] = 1
        param = I - 2 * param
        
        ########################Adaptation
        img_k = self.train_img_k(x) 
        #img_k.shape:   b, 1, self.embedding_dim, 1
        #param_1.shape: p_b, p_l, p_d, p_d
        param = (param*img_k).sum(dim=-2)  #each row is basis
        if model!=None:
            param = model( param )
        param = self.train_mlp(param)
        
        param = self.train_mlp2(self.gelu(param))

        out=F.conv2d(x, param.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding)
        
        return out 
class NAFBlock_modified(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        
        
        # self.middle_dim=dim*3
        dim=dw_channel
        self.middle_dim=32
        self.embedding_dim=32
        
        self.train_basis_conv_en_1 = basis_based_conv(in_channels=dim, 
                                                  out_channels=self.middle_dim,
                                                  embedding_dim= self.embedding_dim,
                                                  kernel_size=1)
        self.train_basis_conv_de_1 = basis_based_conv(in_channels=self.middle_dim, 
                                                  out_channels=dim,
                                                  embedding_dim= self.embedding_dim,
                                                  kernel_size=1)
        
        
        dim_2=ffn_channel
        self.middle_dim_2=32
        self.embedding_dim_2=32
        
        self.train_basis_conv_en_2 = basis_based_conv(in_channels=dim_2, 
                                                  out_channels=self.middle_dim_2,
                                                  embedding_dim= self.embedding_dim_2,
                                                  kernel_size=1)
        self.train_basis_conv_de_2 = basis_based_conv(in_channels=self.middle_dim_2, 
                                                  out_channels=dim_2,
                                                  embedding_dim= self.embedding_dim_2,
                                                  kernel_size=1)

    def forward(self, inp, model=None):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        
        ################################################################################################################
        qkv=self.train_basis_conv_en_1(x)
        qkv=self.train_basis_conv_de_1(qkv)
        ################################################################################################################
        
        # print(x.shape,qkv.shape)
        x = self.sg(x+qkv)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        
        ################################################################################################################
        qkv_2=self.train_basis_conv_en_2(x)
        qkv_2=self.train_basis_conv_de_2(qkv_2)
        ################################################################################################################
        x = self.sg(x+qkv_2)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class NAFNet(nn.Module):

    def __init__(self, 
                img_channel=3, 
                width=16, 
                middle_blk_num=1, 
                enc_blk_nums=[], 
                dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock_modified(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            for block in decoder:
                x = block(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


def sidd_eval(irmodel,noisepath='ValidationNoisyBlocksSrgb.mat',cleanpath='ValidationGtBlocksSrgb.mat'):
    import scipy.io as sio
    filepath = noisepath
    img = sio.loadmat(filepath)
    Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
    Inoisy /=255.
    restored = np.zeros_like(Inoisy)
    with torch.no_grad():
        for i in range(40):
            for k in range(32):
                noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
                restored_patch = irmodel(noisy_patch)
                restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
                restored[i,k,:,:,:] = restored_patch
    filepath = cleanpath
    img = sio.loadmat(filepath)
    Iclean = np.float32(np.array(img['ValidationGtBlocksSrgb']))
    avg_rgb_psnr=[]
    for i in range(40):
        for k in range(32):

            current_psnr =  psnr(restored[i,k,:,:,:]*255.0, Iclean[i,k,:,:,:])
            avg_rgb_psnr.append(current_psnr)
    print(np.mean(avg_rgb_psnr))
    
def psnr(img1,img2):
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))    
    
if __name__ == "__main__":

    model = NAFNet(img_channel=3, 
                            width=64, 
                            middle_blk_num=12, 
                            enc_blk_nums=[2, 2, 4, 8], 
                            # dec_blk_nums=[2, 2, 4, 8],
                            dec_blk_nums=[2, 2, 2, 2]
                            ).to('cuda')
    # # 加载
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key].copy_(torch.load(f'./NAFNet_layer_save/{key.replace(".", "_")}.pt'))
    sidd_eval(model,noisepath='ValidationNoisyBlocksSrgb.mat',cleanpath='ValidationGtBlocksSrgb.mat')
    