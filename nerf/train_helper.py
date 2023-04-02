from nerf import models

densitytype = 'NeRFModelDensitymodule'
colortype = 'NeRFModelColormodule'
nerftype = 'NerfHY'


def create_part(cfg, device, no_coarse_color=False):
    model_coarse_density = getattr(models, densitytype)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
        num_layers=cfg.models.coarse.num_layers,
        hidden_size=cfg.models.coarse.hidden_size,
        include_expression=True
    )
    model_coarse_density.to(device)

    if no_coarse_color:
        model_coarse_color = None
    else:
        model_coarse_color = getattr(models, colortype)(
            num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
            include_input_xyz=cfg.models.coarse.include_input_xyz,
            include_input_dir=cfg.models.coarse.include_input_dir,
            use_viewdirs=cfg.models.coarse.use_viewdirs,
            num_layers=cfg.models.coarse.num_layers,
            hidden_size=cfg.models.coarse.hidden_size,
            include_expression=True
        )
        model_coarse_color.to(device)
    
    # If a fine-resolution model is specified, initialize it.
    if hasattr(cfg.models, "fine"):
        model_fine_density = getattr(models, densitytype)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
            num_layers = cfg.models.coarse.num_layers,
            hidden_size =cfg.models.coarse.hidden_size,
            include_expression=True
        )
        model_fine_color = getattr(models, colortype)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
            num_layers = cfg.models.coarse.num_layers,
            hidden_size =cfg.models.coarse.hidden_size,
            include_expression=True
        )
    
        model_fine_density.to(device)
        model_fine_color.to(device)  
        
    return  model_coarse_density, model_coarse_color, model_fine_density, model_fine_color
    
def create_module(cfg, device, model_coarse_density, model_coarse_color, model_fine_density, model_fine_color, transblock_patch, transblock_cross, fix_density=False): 
    #---------------------- density + color----------------------#
    model_coarse = getattr(models, nerftype)(
        model_coarse_density,
        model_coarse_color,
        fix_density
        )
    model_fine = getattr(models, nerftype)(
        model_fine_density,
        model_fine_color,
        fix_density
        )
    if cfg.models.remove.transformer:
        if cfg.models.remove.single:
            removecolor = getattr(models, cfg.models.remove.type2)(
                transblock_patch,
                transblock_cross,
                input_nc=cfg.models.remove.input_nc, 
                output_nc=cfg.models.remove.output_nc, 
                ngf=cfg.models.remove.ngf
            )
        else:
            # multi scale
            removecolor = getattr(models, cfg.models.remove.type3)(
                transblock_patch,
                transblock_cross,
                input_nc=cfg.models.remove.input_nc, 
                output_nc=cfg.models.remove.output_nc, 
                ngf=cfg.models.remove.ngf
            )
    else:
        removecolor = getattr(models, cfg.models.remove.type)(
            input_nc=cfg.models.remove.input_nc, 
            output_nc=cfg.models.remove.output_nc, 
            ngf=cfg.models.remove.ngf
        )
    removecolor.to(device)
    return model_coarse, model_fine, removecolor