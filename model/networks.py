import functools
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler


####################################################################
#------------------------- Discriminators --------------------------
####################################################################

class Dis_pair(nn.Module):
    def __init__(self, input_dim_a=3, input_dim_b=3, dis_n_layer=5, norm='None', sn=True):
        super(Dis_pair, self).__init__()
        ch = 64
        self.model = self._make_net(ch, input_dim_a+input_dim_b, dis_n_layer, norm, sn)

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]
        tch = ch
        for i in range(1, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]
            tch *= 2
        model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)]
        tch *= 2
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0))]
        else:
            model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]
        return nn.Sequential(*model)

    def forward(self, image_a, image_b):
        out = torch.cat((image_a, image_b), 1)
        out = self.model(out)
        out = out.view(-1)
        outs = []
        outs.append(out)
        return outs

class Dis_single(nn.Module):
    def __init__(self, input_dim=3, dis_n_layer=5, norm='None', sn=True):
        super(Dis_single, self).__init__()
        ch = 64
        self.model = self._make_net(ch, input_dim, dis_n_layer, norm, sn)

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]
        tch = ch
        for i in range(1, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]
            tch *= 2
        model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)]
        tch *= 2
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0))]
        else:
            model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]
        return nn.Sequential(*model)

    def forward(self, image):
        out = self.model(image)
        out = out.view(-1)
        outs = []
        outs.append(out)
        return outs

####################################################################
#------------------------- Basic Functions -------------------------
####################################################################

def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    elif opts.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opts.lr_policy)
    return scheduler

def init_weights(net, init_type, gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fainplanes')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, gpu, init_type='normal', gain=0.02):
    assert(torch.cuda.is_available())
    net.to(gpu)
    init_weights(net, init_type, gain)
    return net

def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    elif opts.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opts.lr_policy)
    return scheduler

####################################################################
#-------------------------- makeup loss --------------------------
####################################################################

def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def to_var( x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    if not requires_grad:
        return Variable(x, requires_grad=requires_grad)
    else:
        return Variable(x)

####################################################################
#-------------------------- Basic Blocks --------------------------
####################################################################

# conv + (spectral) + (instance) + leakyrelu
class LeakyReLUConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(outplanes, affine=False)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class Residule_Block(nn.Module):
    def __init__(self, nc):
        super(Residule_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nc)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nc)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(x)
        out = self.bn2(out)
        return out + x
