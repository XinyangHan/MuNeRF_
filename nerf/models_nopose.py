import torch
import functools

class VeryTinyNeRFModel(torch.nn.Module):
    r"""Define a "very tiny" NeRF model comprising three fully connected layers.
    """

    def __init__(self, filter_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(VeryTinyNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 65 -> 128)
        self.layer1 = torch.nn.Linear(
            self.xyz_encoding_dims + self.viewdir_encoding_dims, filter_size
        )
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class MultiHeadNeRFModel(torch.nn.Module):
    r"""Define a "multi-head" NeRF model (radiance and RGB colors are predicted by
    separate heads).
    """

    def __init__(self, hidden_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(MultiHeadNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 39 -> 128)
        self.layer1 = torch.nn.Linear(self.xyz_encoding_dims, hidden_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 3_1 (default: 128 -> 1): Predicts radiance ("sigma")
        self.layer3_1 = torch.nn.Linear(hidden_size, 1)
        # Layer 3_2 (default: 128 -> 1): Predicts a feature vector (used for color)
        self.layer3_2 = torch.nn.Linear(hidden_size, hidden_size)

        # Layer 4 (default: 39 + 128 -> 128)
        self.layer4 = torch.nn.Linear(
            self.viewdir_encoding_dims + hidden_size, hidden_size
        )
        # Layer 5 (default: 128 -> 128)
        self.layer5 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 6 (default: 128 -> 3): Predicts RGB color
        self.layer6 = torch.nn.Linear(hidden_size, 3)

        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x, view = x[..., : self.xyz_encoding_dims], x[..., self.xyz_encoding_dims :]
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        sigma = self.layer3_1(x)
        feat = self.relu(self.layer3_2(x))
        x = torch.cat((feat, view), dim=-1)
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.layer6(x)
        return torch.cat((x, sigma), dim=-1)


class ReplicateNeRFModel(torch.nn.Module):
    r"""NeRF model that follows the figure (from the supp. material of NeRF) to
    every last detail. (ofc, with some flexibility)
    """

    def __init__(
        self,
        hidden_size=256,
        num_layers=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        **kwargs
    ):
        super(ReplicateNeRFModel, self).__init__()
        # xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions

        self.dim_xyz = (3 if include_input_xyz else 0) + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = (3 if include_input_dir else 0) + 2 * 3 * num_encoding_fn_dir

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_alpha = torch.nn.Linear(hidden_size, 1)

        self.layer4 = torch.nn.Linear(hidden_size + self.dim_dir, hidden_size // 2)
        self.layer5 = torch.nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, direction = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        x_ = self.relu(self.layer1(xyz))
        x_ = self.relu(self.layer2(x_))
        feat = self.layer3(x_)
        alpha = self.fc_alpha(x_)
        y_ = self.relu(self.layer4(torch.cat((feat, direction), dim=-1)))
        y_ = self.relu(self.layer5(y_))
        rgb = self.fc_rgb(y_)
        return torch.cat((rgb, alpha), dim=-1)


class PaperNeRFModel(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
    ):
        super(PaperNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        

        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz, 256))
        for i in range(1, 6):
            if i == 3:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        x = xyz#self.relu(self.layers_xyz[0](xyz))
        for i in range(6):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)




class ConditionalBlendshapePaperNeRFModel(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        include_expression=True,
        latent_code_dim=32

    ):
        super(ConditionalBlendshapePaperNeRFModel, self).__init__()
        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 79 if include_expression else 0
        
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz # 3+2*3*6=39
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir # 3+2*3*4=27
        self.dim_expression = include_expression# + 2 * 3 * num_encoding_fn_expr
        self.dim_latent_code = latent_code_dim

        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        #print('!!!!!!', self.dim_xyz + self.dim_expression + self.dim_latent_code + self.dim_lights)
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code , 256)) # + self.dim_details + self.dim_lights
        for i in range(1, 6):
            if i == 3:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code + 256, 256)) # + self.dim_details  + self.dim_lights
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x,  expr=None, latent_code=None, **kwargs):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        x = xyz#self.relu(self.layers_xyz[0](xyz))
        latent_code = latent_code.repeat(xyz.shape[0], 1)
        if self.dim_expression > 0:
            expr_encoding = (expr * 1 / 3).repeat(xyz.shape[0], 1)
            #torch.Size([2048, 63]) torch.Size([2048, 50]) torch.Size([2048, 32]) torch.Size([2048, 9])
            initial = torch.cat((xyz, expr_encoding, latent_code), dim=1) # detail_encoding
            x = initial # torch.Size([2048, 154])
        for i in range(6):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((initial, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x) #([2048, 256])
        alpha = self.fc_alpha(feat) #([2048, 1])
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)  #([2048, 3])
        return torch.cat((rgb, alpha), dim=-1)

class NeRFModelColormodule(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        include_expression=True,
        latent_code_dim=32

    ):
        super(NeRFModelColormodule, self).__init__()
        include_lights = True
        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 79 if include_expression else 0
        
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz # 3+2*3*6=39
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir # 3+2*3*4=27
        self.dim_expression = include_expression# + 2 * 3 * num_encoding_fn_expr
        self.dim_latent_code = latent_code_dim
        self.use_viewdirs = use_viewdirs

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, feat, x, expr=None, latent_code=None, **kwargs):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)  #([2048, 3])
        return rgb    

class NeRFModelDensitymodule(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        include_expression=True,
        latent_code_dim=32

    ):
        super(NeRFModelDensitymodule, self).__init__()
        include_lights = True
        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 79 if include_expression else 0
        
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz # 3+2*3*6=39
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir # 3+2*3*4=27
        self.dim_expression = include_expression# + 2 * 3 * num_encoding_fn_expr
        self.dim_latent_code = latent_code_dim

        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code, 256)) # + self.dim_details
        for i in range(1, 6):
            if i == 3:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code + 256, 256)) # + self.dim_details 
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)
        self.relu = torch.nn.functional.relu
    
    def forward(self, x,  expr=None, latent_code=None, **kwargs):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        x = xyz#self.relu(self.layers_xyz[0](xyz))
        latent_code = latent_code.repeat(xyz.shape[0], 1)
        if self.dim_expression > 0:
            expr_encoding = (expr * 1 / 3).repeat(xyz.shape[0], 1)
            initial = torch.cat((xyz, expr_encoding, latent_code), dim=1) # detail_encoding
            x = initial # torch.Size([2048, 154])
        for i in range(6):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((initial, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat) #([2048, 1])
        
        return feat, alpha

class NerfHY(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(
        self,
        density, 
        color,
        fix_density
    ):
        super(NerfHY, self).__init__()

        self.density = density
        self.color = color
        self.fix_density = fix_density
        
    def forward(self, x, expr=None, latent_code=None, **kwargs):
        if self.fix_density:
            with torch.no_grad():
                feat, alpha = self.density(x, expr, latent_code, **kwargs)
        else:
            feat, alpha = self.density(x, expr, latent_code, **kwargs)
        # feat, alpha = self.density(x, expr, latent_code, **kwargs)
        if self.color is not None:
            rgb = self.color(feat, x, expr, latent_code, **kwargs)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return alpha

class ResnetGenerator(torch.nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, 
        input_nc, 
        output_nc, 
        ngf=64, 
        norm_layer=torch.nn.BatchNorm2d, 
        use_dropout=True, 
        n_blocks=9, 
        padding_type='reflect'):

        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == torch.nn.InstanceNorm2d
        else:
            use_bias = norm_layer == torch.nn.InstanceNorm2d

        model = [torch.nn.ReflectionPad2d(3),
                 torch.nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 torch.nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [torch.nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      torch.nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [torch.nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      torch.nn.ReLU(True)]
        model += [torch.nn.ReflectionPad2d(3)]
        model += [torch.nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [torch.nn.Tanh()]

        self.model = torch.nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetGenerator_upsample(torch.nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, 
        input_nc, 
        output_nc, 
        ngf=64, 
        norm_layer=torch.nn.BatchNorm2d, 
        use_dropout=True, 
        n_blocks=9, 
        padding_type='reflect'):

        assert(n_blocks >= 0)
        super(ResnetGenerator_upsample, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == torch.nn.InstanceNorm2d
        else:
            use_bias = norm_layer == torch.nn.InstanceNorm2d

        model = [torch.nn.ReflectionPad2d(3),
                 torch.nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 torch.nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [torch.nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      torch.nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [torch.nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      torch.nn.ReLU(True)]

        for i in range(n_downsampling):  # add upsampling layers
            # mult = 2 ** (n_downsampling - i)
            model += [torch.nn.ConvTranspose2d(ngf, ngf,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(ngf),
                      torch.nn.ReLU(True)]
        
        model += [torch.nn.ReflectionPad2d(3)]
        model += [torch.nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [torch.nn.Tanh()]

        self.model = torch.nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetGenerator_newupsample(torch.nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, 
        input_nc, 
        output_nc, 
        ngf=64, 
        norm_layer=torch.nn.BatchNorm2d, 
        use_dropout=True, 
        n_blocks=9, 
        padding_type='reflect'):

        assert(n_blocks >= 0)
        super(ResnetGenerator_newupsample, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == torch.nn.InstanceNorm2d
        else:
            use_bias = norm_layer == torch.nn.InstanceNorm2d
        # ------ 1*1 ---------*
        res_model = []

        res_model = [torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            torch.nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            res_model += [torch.nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=7, padding=0, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      torch.nn.ReLU(True)]

        # ------ resnet ---------*
        n_upsampling = 2
        mult = 2 ** n_upsampling
        for i in range(n_blocks):       # add ResNet blocks
            res_model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        # ------ upsample ---------*
        upsample_model1 = []
        for i in range(n_upsampling):  # add upsampling layers
            mult = 2 ** (n_upsampling - i) # 2 1 
            upsample_model1 += [torch.nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=7, stride=1,
                                         padding=0, output_padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      torch.nn.ReLU(True)]
        
        #for i in range(n_downsampling):  # add upsampling layers
        upsample_model2 = [torch.nn.ConvTranspose2d(ngf*5, ngf,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(ngf),
                    torch.nn.ReLU(True)]
        upsample_model3 = [torch.nn.ConvTranspose2d(ngf*3, ngf,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(ngf),
                    torch.nn.ReLU(True)]

        # ------ 1*1 ---------*
        conv1_model = []
        conv1_model += [torch.nn.ReflectionPad2d(3)]
        conv1_model += [torch.nn.Conv2d(ngf*2, ngf, kernel_size=7, padding=0)]
        conv1_model += [torch.nn.ReLU(True)]
        conv1_model += [torch.nn.ReflectionPad2d(3)]
        conv1_model += [torch.nn.Conv2d(ngf, ngf, kernel_size=7, padding=0)]
        conv1_model += [torch.nn.ReLU(True)]
        conv1_model += [torch.nn.ReflectionPad2d(3)]
        conv1_model += [torch.nn.Conv2d(ngf, 3, kernel_size=7, padding=0)]
        conv1_model += [torch.nn.Tanh()]

        # --- global ---
        model_g = [torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            torch.nn.ReLU(True)] #128

        mult = 1
        modelg_1 = [torch.nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                torch.nn.ReLU(True)] #64
        model_g_128 = model_g
        mult = 2
        modelg_2 = [torch.nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                torch.nn.ReLU(True)]#32
        mult = 4
        self.model_g128 =torch.nn.Sequential(*model_g_128)
        self.model_g64 =torch.nn.Sequential(*modelg_1)
        self.model_g32 =torch.nn.Sequential(*modelg_2)
        self.res_model = torch.nn.Sequential(*res_model)
        self.upsample_model1 = torch.nn.Sequential(*upsample_model1)
        self.upsample_model2 = torch.nn.Sequential(*upsample_model2)
        self.upsample_model3 = torch.nn.Sequential(*upsample_model3)
        self.conv1_model = torch.nn.Sequential(*conv1_model)

    def forward(self, input, global_input):
        # global input
        global_128 = self.model_g128(global_input)
        global_64 = self.model_g64(global_128)
        global_32 = self.model_g32(global_64)

        """Standard forward"""
        res_x = self.res_model(input)
        upsample_x = self.upsample_model1(res_x)  # 1, 64, 32, 32
        upsample_x_cat_g32 = torch.cat((upsample_x, global_32), dim=1)
        upsample_x64 = self.upsample_model2(upsample_x_cat_g32) 
        upsample_x_cat_g64 = torch.cat((upsample_x64, global_64), dim=1)
        upsample_x128 = self.upsample_model3(upsample_x_cat_g64) 
        upsample_x_cat_g128 = torch.cat((upsample_x128, global_128), dim=1)
        x_out = self.conv1_model(upsample_x_cat_g128)
        return x_out

class ResnetGenerator_newupsample2(torch.nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, 
        transblock_patch,
        transblock_cross,
        input_nc, 
        output_nc, 
        ngf=64, 
        ngf_= 50,
        norm_layer=torch.nn.BatchNorm2d, 
        use_dropout=True, 
        n_blocks=9, 
        padding_type='reflect'):

        assert(n_blocks >= 0)
        super(ResnetGenerator_newupsample2, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == torch.nn.InstanceNorm2d
        else:
            use_bias = norm_layer == torch.nn.InstanceNorm2d
        # ------ 1*1 ---------*
        res_model = []

        res_model = [torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            torch.nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            res_model += [torch.nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=7, padding=0, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      torch.nn.ReLU(True)]

        # ----- conv for transformer ----*
        conv_model = []
        conv_model = [torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_nc*2, int(ngf/4), kernel_size=2, stride=2, padding=4, bias=use_bias),
            norm_layer(int(ngf/4)),
            torch.nn.ReLU(True)]

        # ------ resnet ---------*
        n_upsampling = 2
        mult = 2 ** n_upsampling
        for i in range(n_blocks):       # add ResNet blocks
            res_model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        # ------ upsample ---------*
        upsample_model1 = []
        for i in range(n_upsampling):  # add upsampling layers
            mult = 2 ** (n_upsampling - i) # 2 1 
            upsample_model1 += [torch.nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=7, stride=1,
                                         padding=0, output_padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      torch.nn.ReLU(True)]
        
        #for i in range(n_downsampling):  # add upsampling layers
        upsample_model2 = [torch.nn.ConvTranspose2d(464, ngf,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(ngf),
                    torch.nn.ReLU(True)]
        upsample_model3 = [torch.nn.ConvTranspose2d(ngf*5, ngf,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(ngf),
                    torch.nn.ReLU(True)]
        upsample_model4 = [torch.nn.ConvTranspose2d(ngf*3, ngf,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(ngf),
                    torch.nn.ReLU(True)]

        # ------ 1*1 ---------*
        conv1_model = []
        conv1_model += [torch.nn.ReflectionPad2d(3)]
        conv1_model += [torch.nn.Conv2d(ngf*2, ngf, kernel_size=7, padding=0)]
        conv1_model += [torch.nn.ReLU(True)]
        conv1_model += [torch.nn.ReflectionPad2d(3)]
        conv1_model += [torch.nn.Conv2d(ngf, ngf, kernel_size=7, padding=0)]
        conv1_model += [torch.nn.ReLU(True)]
        conv1_model += [torch.nn.ReflectionPad2d(3)]
        conv1_model += [torch.nn.Conv2d(ngf, 3, kernel_size=7, padding=0)]
        conv1_model += [torch.nn.Tanh()]

        # --- global ---
        model_g = [torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            torch.nn.ReLU(True)] #128

        mult = 1
        modelg_1 = [torch.nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                torch.nn.ReLU(True)] #64
        model_g_128 = model_g
        mult = 2
        modelg_2 = [torch.nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                torch.nn.ReLU(True)]#32
        mult = 4
        modelg_2_ = [torch.nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult),
                torch.nn.ReLU(True)]#32
        self.model_g128 =torch.nn.Sequential(*model_g_128)
        self.model_g64 =torch.nn.Sequential(*modelg_1)
        self.model_g32 =torch.nn.Sequential(*modelg_2)
        self.model_g32_ = torch.nn.Sequential(*modelg_2_)
        self.res_model = torch.nn.Sequential(*res_model)
        self.upsample_model1 = torch.nn.Sequential(*upsample_model1)
        self.upsample_model2 = torch.nn.Sequential(*upsample_model2)
        self.upsample_model3 = torch.nn.Sequential(*upsample_model3)
        self.upsample_model4 = torch.nn.Sequential(*upsample_model4)
        self.conv1_model = torch.nn.Sequential(*conv1_model)
        self.patch_transformer = transblock_patch
        self.cross_transformer = transblock_cross
        self.conv_patch_pair = torch.nn.Sequential(*conv_model)

    def forward(self, input, global_input, input_pairs, pose, cross):
        # global input
        if global_input.shape[3]==256:
        # print('debug!,' ,global_input.shape)
            flag=1
            global_256 = self.model_g128(global_input)
            global_128 = self.model_g64(global_256)
            global_64 = self.model_g32(global_128)
            global_32 = self.model_g32_(global_64)
        else:
            flag=0
            global_128 = self.model_g128(global_input)
            global_64 = self.model_g64(global_128)
            global_32 = self.model_g32(global_64)

        # crop patches referto patchgan: 0: content 1: style
        patch_50_contents = input_pairs[0]
        patch_50_styles = input_pairs[1]
        patch_feat = []
        patch_all_cat = []
        for patch_c, patch_s in zip(patch_50_contents,patch_50_styles):
           patch_all_cat.append(torch.cat((patch_c,patch_s), dim=1))
        patch_trans_cross = torch.cat(tuple(patch_all_cat),dim=1)
        for patch_c, patch_s in zip(patch_50_contents,patch_50_styles):
            patch_cs = torch.cat((patch_c,patch_s), dim=1)
            #conv_patch_cs = self.conv_patch_pair(patch_cs)
            trans_patch_cs =  self.patch_transformer(patch_cs)
            patch_feat.append(trans_patch_cs)
        patch_trans_feat = torch.cat(tuple(patch_feat),dim=1)# 64*12
        if cross:
            # print('use cross patch')
            patch_trans_feat_ = self.cross_transformer(patch_trans_cross)
            patch_trans_feat_all = torch.cat((patch_trans_feat_,patch_trans_feat),dim=1)
        """Standard forward"""
        res_x = self.res_model(input)
        upsample_x = self.upsample_model1(res_x)  # 1, 64, 32, 32
        upsample_x_cat_g32 = torch.cat((patch_trans_feat_all, upsample_x, global_32), dim=1)
        upsample_x64 = self.upsample_model2(upsample_x_cat_g32) 
        upsample_x_cat_g64 = torch.cat((upsample_x64, global_64), dim=1)
        upsample_x128 = self.upsample_model3(upsample_x_cat_g64) 
        upsample_x_cat_g128 = torch.cat((upsample_x128, global_128), dim=1)
        if flag==0:
            x_out = self.conv1_model(upsample_x_cat_g128)
        else:
            upsample_x256 = self.upsample_model4(upsample_x_cat_g128)
            #upsample_x_cat_g256 = upsample_x256
            upsample_x_cat_g256 = torch.cat((upsample_x256, global_256), dim=1)
            x_out = self.conv1_model(upsample_x_cat_g256)
        return x_out

class ResnetGenerator_newupsample3(torch.nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, 
        transblock_patch,
        transblock_cross,
        input_nc, 
        output_nc, 
        ngf=64, 
        ngf_= 50,
        norm_layer=torch.nn.BatchNorm2d, 
        use_dropout=True, 
        n_blocks=9, 
        padding_type='reflect'):

        assert(n_blocks >= 0)
        super(ResnetGenerator_newupsample3, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == torch.nn.InstanceNorm2d
        else:
            use_bias = norm_layer == torch.nn.InstanceNorm2d
        # ------ 1*1 ---------*
        res_model = []

        res_model = [torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            torch.nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            res_model += [torch.nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=7, padding=0, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      torch.nn.ReLU(True)]

        # ----- conv for transformer ----*
        conv_model1 = []
        conv_model1 = [torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_nc*2, int(ngf/4), kernel_size=2, stride=2, padding=4, bias=use_bias),
            norm_layer(int(ngf/4)),
            torch.nn.ReLU(True)]
        conv_model2 = []
        conv_model2= [torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_nc*2, int(ngf/4), kernel_size=2, stride=2, padding=4, bias=use_bias),
            norm_layer(int(ngf/4)),
            torch.nn.ReLU(True)]
        conv_model3 = []
        conv_model3 = [torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_nc*2, int(ngf/4), kernel_size=3, stride=1, padding=2, bias=use_bias),
            norm_layer(int(ngf/4)),
            torch.nn.ReLU(True)]

        # ------ resnet ---------*
        n_upsampling = 2
        mult = 2 ** n_upsampling
        for i in range(n_blocks):       # add ResNet blocks
            res_model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        # ------ upsample ---------*
        upsample_model1 = []
        for i in range(n_upsampling):  # add upsampling layers
            mult = 2 ** (n_upsampling - i) # 2 1 
            upsample_model1 += [torch.nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=7, stride=1,
                                         padding=0, output_padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      torch.nn.ReLU(True)]
        
        #for i in range(n_downsampling):  # add upsampling layers
        upsample_model2 = [torch.nn.ConvTranspose2d(ngf*32, ngf,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(ngf),
                    torch.nn.ReLU(True)]
        upsample_model3 = [torch.nn.ConvTranspose2d(ngf*3, ngf,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(ngf),
                    torch.nn.ReLU(True)]

        # ------ 1*1 ---------*
        conv1_model = []
        conv1_model += [torch.nn.ReflectionPad2d(3)]
        conv1_model += [torch.nn.Conv2d(ngf*2, ngf, kernel_size=7, padding=0)]
        conv1_model += [torch.nn.ReLU(True)]
        conv1_model += [torch.nn.ReflectionPad2d(3)]
        conv1_model += [torch.nn.Conv2d(ngf, ngf, kernel_size=7, padding=0)]
        conv1_model += [torch.nn.ReLU(True)]
        conv1_model += [torch.nn.ReflectionPad2d(3)]
        conv1_model += [torch.nn.Conv2d(ngf, 3, kernel_size=7, padding=0)]
        conv1_model += [torch.nn.Tanh()]

        # --- global ---
        model_g = [torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            torch.nn.ReLU(True)] #128

        mult = 1
        modelg_1 = [torch.nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                torch.nn.ReLU(True)] #64
        model_g_128 = model_g
        mult = 2
        modelg_2 = [torch.nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                torch.nn.ReLU(True)]#32
        mult = 4
        self.model_g128 =torch.nn.Sequential(*model_g_128)
        self.model_g64 =torch.nn.Sequential(*modelg_1)
        self.model_g32 =torch.nn.Sequential(*modelg_2)
        self.res_model = torch.nn.Sequential(*res_model)
        self.upsample_model1 = torch.nn.Sequential(*upsample_model1)
        self.upsample_model2 = torch.nn.Sequential(*upsample_model2)
        self.upsample_model3 = torch.nn.Sequential(*upsample_model3)
        self.conv1_model = torch.nn.Sequential(*conv1_model)
        self.patch_transformer = transblock_patch #[transblock_patch, transblock_patch1, transblock_patch2]
        self.cross_transformer = transblock_cross #[transblock_cross, transblock_cross1, transblock_cross2]
        #self.conv_patch_pair = [torch.nn.Sequential(*conv_model1), torch.nn.Sequential(*conv_model2), torch.nn.Sequential(*conv_model3)]
        self.conv_patch_pair1 = torch.nn.Sequential(*conv_model1)
        self.conv_patch_pair2 = torch.nn.Sequential(*conv_model2)
        self.conv_patch_pair3 = torch.nn.Sequential(*conv_model3)

    def forward(self, input, global_input, input_pairs, cross):
        # global input
        global_128 = self.model_g128(global_input)
        global_64 = self.model_g64(global_128)
        global_32 = self.model_g32(global_64)

        # crop patches referto patchgan: 0: content 1: style
        patch_feat_all = []
        for i in range(3):
            patch_feat = []
            patch_content, patch_style = input_pairs[0][i], input_pairs[1][i]
            for patch_c, patch_s in zip(patch_content, patch_style):
                patch_cs = torch.cat((patch_c, patch_s), dim=1)
            #    print('shape!!!!', patch_cs.shape)
                conv_patch_cs = self.conv_patch_pair1(patch_cs)
            #    print('after conv', conv_patch_cs.shape)
                trans_patch_cs = self.patch_transformer(conv_patch_cs)
            #    print('after trans', trans_patch_cs.shape)
                patch_feat.append(trans_patch_cs)
            patch_trans_feat = torch.cat(tuple(patch_feat),dim=1)
            if cross:
                patch_trans_feat_ = self.cross_transformer(patch_trans_feat)
                patch_trans_feat_all = torch.cat((patch_trans_feat_,patch_trans_feat),dim=1)
                patch_feat_all.append(patch_trans_feat_all)
            patch_feat_all.append(patch_trans_feat)
        patch_feat_all = torch.cat(tuple(patch_feat_all),dim=1)
       
        """Standard forward"""
        res_x = self.res_model(input)
        upsample_x = self.upsample_model1(res_x)  # 1, 64, 32, 32
        upsample_x_cat_g32 = torch.cat((patch_feat_all, upsample_x, global_32), dim=1)
        upsample_x64 = self.upsample_model2(upsample_x_cat_g32) 
        upsample_x_cat_g64 = torch.cat((upsample_x64, global_64), dim=1)
        upsample_x128 = self.upsample_model3(upsample_x_cat_g64) 
        upsample_x_cat_g128 = torch.cat((upsample_x128, global_128), dim=1)
        x_out = self.conv1_model(upsample_x_cat_g128)
        return x_out

class myupsample(torch.nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, 
        input_nc, 
        output_nc, 
        ngf=64, 
        norm_layer=torch.nn.InstanceNorm2d, 
        use_dropout=True, 
        n_blocks=2):

        assert(n_blocks >= 0)
        super(myupsample, self).__init__()
        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == torch.nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == torch.nn.InstanceNorm2d
        use_bias = False

        current_channel = 3 if input_nc == 3 else ngf
        model = [torch.nn.ConvTranspose2d(input_nc, current_channel,
                                     kernel_size=4, stride=2, padding=1,
                                     bias=use_bias),
                  norm_layer(current_channel),
                  torch.nn.LeakyReLU(0.2, inplace=True)]
        # for i in range(n_blocks-2):
        #     model += [torch.nn.ConvTranspose2d(ngf, ngf,
        #                                  kernel_size=4, stride=2, padding=1,
        #                                  bias=use_bias),
        #               norm_layer(ngf),
        #               torch.nn.LeakyReLU(0.2, inplace=True)]
        
        model += [torch.nn.ConvTranspose2d(current_channel, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        model += [torch.nn.Tanh()]

        self.model = torch.nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(torch.nn.Module):
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
            conv_block += [torch.nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [torch.nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [torch.nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), torch.nn.ReLU(True)]
        if use_dropout:
            conv_block += [torch.nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [torch.nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [torch.nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [torch.nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return torch.nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class DiscriminatorModel(torch.nn.Module):
    def __init__(self, dim_latent=32, dim_expressions=76):
        super(DiscriminatorModel, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim_latent, dim_latent*2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(dim_latent*2, dim_latent*2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(dim_latent*2, dim_expressions),
            torch.nn.Tanh(),
        )

    def forward(self, x):

        return self.model(x)
