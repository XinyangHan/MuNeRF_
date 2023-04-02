import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from model.transEncoder import Block
from torchvision import transforms
import PIL

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'SwitchableNorm':
        norm_layer = functools.partial(nn.SwitchableNorm, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(use_resnet, if_global, input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'TransEncoder':
        net = ResnetGeneratorTransGenerator(256, use_resnet, if_global, input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)



##############################################################################
# Classes
##############################################################################

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetGeneratorTransGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, img_size, use_resnet, if_global, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGeneratorTransGenerator, self).__init__()
        #print('n_blocks@@@@@@@@@@@@@@@@',n_blocks)
        # transformer  
        self.img_size = img_size
        self.times = self.img_size // 32 #256/32=8
        self.embed_dim = 16*(self.times ** 2)
        self.embed_dim2 = 4*(self.times ** 2)
        depth=2
        num_heads=4
        mlp_ratio=4.
        qkv_bias=False
        qk_scale=None
        drop_rate=0.
        attn_drop_rate=0.
        drop_path_rate=0.
        hybrid_backbone=None
        norm_layer1=nn.LayerNorm
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # embed_dim divide num_heads
        self.TransformerBlock = nn.ModuleList([
                Block(
                    dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer1)
            for i in range(depth)])
        
        self.TransformerBlock2 = nn.ModuleList([
                Block(
                    dim=self.embed_dim2, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer1)
            for i in range(depth)])
        
        
        #

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        

        model_p = [#nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=2, stride=2, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]           
                 
                 
        model_g = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        
        model = []
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        
        modelconv = [nn.ReflectionPad2d(3),
                 nn.Conv2d(256, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        
        model_p1 = model_p
        
        modelp = model_p + model + modelconv  # 1, 256, 32,32  # conv part is the front
        mult = 1
        modelg_1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)]
        mult = 2
        modelg_2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)]

     # deconv part is the back
     
     #****************back part: global and patch& only patch*******************#   
        #global and patch 
        if if_global == 'True':
            #style global transformer
            #k=3
            k = 1
        else:
            k = 1
        
        model = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult*k, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        if if_global == 'True':
            channel_scale = 8
        else:
            channel_scale = 4 
        # ----------------------------------------model conv -----------------------------------#
        n_resnet = int(n_blocks/2)
        model_conv = [nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf*channel_scale, ngf, kernel_size=3, padding=1, bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]
        if use_resnet == 'True':
            for i in range(int(n_resnet)):
                model_conv += [ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_conv = nn.Sequential(*model_conv)
        # ----------------------------------------model convs -----------------------------------#
        if if_global == 'True':
            #style global transformer
            k1 = 8
            k2 = 10
            #k1 = 12
            #k2 = 6
        else:
            k1 = 8
            k2 = 6
        model_conv2 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(ngf*k1, ngf*channel_scale, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)] 
        if use_resnet == 'True':
            for i in range(int(n_resnet)):
                model_conv2 += [ResnetBlock(ngf*channel_scale, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_conv2 = nn.Sequential(*model_conv2)

        model_conv3 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(ngf*k2, ngf*channel_scale, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)] 
        if use_resnet == 'True':
            for i in range(int(n_resnet)):
                model_conv3 += [ResnetBlock(ngf*channel_scale, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_conv3 = nn.Sequential(*model_conv3)
        # -----------------------------------------modelconv_g----------------------------------# 
        modelconv_g = []
        modelconv_g += [nn.ReflectionPad2d(3),
                 nn.Conv2d(ngf*2, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)] 
        
        # -----------------------------------------model 3 -------------------------------------# 
        model3 = []
        n_upsampling = 2
        ngf =  ngf*2 
        for i in range(n_upsampling):
            mult = (2**i)
            ngf_i = int(ngf/mult)
            ngf_o = int(ngf_i/2)
            print(mult, ngf, ngf_o)
            model3 += [nn.ReflectionPad2d(3)]
            model3 += [nn.Conv2d(ngf_i, ngf_o, kernel_size=7, padding=0)]
            model3 += [nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(32, 3, kernel_size=7, padding=0)]
        
        
        #if use_resnet == 'True':
        #    for i in range(int(n_resnet)):
        #        model3 += [ResnetBlock(3, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
        model3 += [nn.Tanh()]
        # for conv part 
        self.modelp = nn.Sequential(*modelp)
        self.modelp1 = nn.Sequential(*model_p1)
        self.modelg = nn.Sequential(*model_g)
        self.modelg_1 = nn.Sequential(*modelg_1)
        self.modelg_2 = nn.Sequential(*modelg_2)
        #global and patch 
        self.model2 = nn.Sequential(*model)
        self.model3 = nn.Sequential(*model3)
        self.modelconv_g = nn.Sequential(*modelconv_g)

    def forward(self, content_in ,patch1_in, patch2_in, patch3_in, patch4_in, gpu, kernel, if_DCM, if_need, if_global, use_resnet):
        #self.real_A_patch , self.real_A, self.real_A_patch2,
        """Standard forward"""
        dim = 1
        global_ = self.modelg(content_in) #conv
        global_128 = self.modelg_1(global_) #downsample1 x_2
        global_64= self.modelg_2(global_128) #downsample2  y

        patch1 = self.modelp(patch1_in) #patch1
        patch2 = self.modelp(patch2_in) #patch2
        patch3 = self.modelp(patch3_in) #patch1
        patch4 = self.modelp(patch4_in) #patch2
        
        global1 = self.modelp(content_in) #content
        #global2 = self.modelp(style_in) #style global

        if if_need=='True':
            # plan (1) : T(cat(p,g))
            x_pre1 = patch1.view(1,64,16*(self.times ** 2))
            x_pre2 = patch2.view(1,64,16*(self.times ** 2))
            x_pre3 = patch3.view(1,64,16*(self.times ** 2))
            x_pre4 = patch4.view(1,64,16*(self.times ** 2))
            
            for index, blk in enumerate(self.TransformerBlock):
                x_pre1 = blk(x_pre1)
            for index, blk in enumerate(self.TransformerBlock):
                x_pre2 = blk(x_pre2)
            for index, blk in enumerate(self.TransformerBlock):
                x_pre3 = blk(x_pre3)
            for index, blk in enumerate(self.TransformerBlock):
                x_pre4 = blk(x_pre4)
            x_pre = torch.cat((x_pre1,x_pre2,x_pre3,x_pre4),dim)
            x_pre = x_pre.view(-1,64*4,16*(self.times ** 2)) #64*64
            for index, blk in enumerate(self.TransformerBlock):
                x_pre = blk(x_pre)
            x_pre = x_pre.view(-1,64*4, 4*self.times, 4*self.times)   #[1, 512, 32, 32])
            #x_pre = torch.cat((patch1,patch2,patch3,patch4),dim) 
            x = nn.Upsample(64,mode='bilinear')(x_pre)
        #x = torch.cat((patch1_,patch2_),dim)
        x = self.model2(x) 
        if use_resnet == 'False':
            if if_global=='True':
                #x = torch.cat((x , global_64, patch1_64, patch2_64),dim)  #1280
                x = torch.cat((x , global_64),dim)
                x = self.model_conv2(x)
                x = nn.Upsample(128,mode='bilinear')(x)
                x = torch.cat((x, global_128),dim)
                #x = torch.cat((x, global_128),dim)
                x = self.model_conv3(x)
            else:
                x = nn.Upsample(128,mode='bilinear')(x)
        else:
            x = nn.Upsample(128,mode='bilinear')(x)
        #print(f'upsample  {x.shape}')
        x = self.model_conv(x)
        #print(f'model_conv {x.shape}')
        #global_s = self.modelconv_g(torch.cat((patch1_,global2_),dim))
        if if_DCM:
            global_ori = global_
            global_ = DCM(64,64,kernel).to(gpu)(global_,global_)
            #x = torch.cat((nn.Upsample(256,mode='bilinear')(x), global_, global_ori), dim)
            x = torch.cat((nn.Upsample(256,mode='bilinear')(x),global_),dim)
        else:     
            x = torch.cat((nn.Upsample(256,mode='bilinear')(x),global_),dim)
        #x = nn.Upsample(256,mode='bilinear')(x)
        #print(f'upsample {x.shape}')
        #x = DCM(128,3).to(gpu)(x,x)
        x = self.model3(x) #[1,128,256,256]
        #print(f'model3  {x.shape}')
        return x

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out