'''
Copy the credentials provided by Face++ below and change the name of this file to settings.py
'''

FACEPPAPI_KEY = "wGR3QfaPASvV0RZJxum12Cu75sqNctPZ"
FACEPPAPI_SCRECT = "7P3dpVyecaryXKn2aVe5gETf_cfVmmw6"
DENSITY_TRAIN = ['layers_xyz.0.weight', 'layers_xyz.0.bias', 
                'layers_xyz.1.weight', 'layers_xyz.1.bias', 
                'layers_xyz.2.weight', 'layers_xyz.2.bias', 
                'layers_xyz.3.weight', 'layers_xyz.3.bias', 
                'layers_xyz.4.weight', 'layers_xyz.4.bias', 
                'layers_xyz.5.weight', 'layers_xyz.5.bias', 
                'fc_feat.weight', 'fc_feat.bias', 
                'fc_alpha.weight', 'fc_alpha.bias']
COLOR_TRAIN  = ['layers_dir.0.weight', 'layers_dir.0.bias', 
                'layers_dir.1.weight', 'layers_dir.1.bias', 
                'layers_dir.2.weight', 'layers_dir.2.bias', 
                'layers_dir.3.weight', 'layers_dir.3.bias', 
                'fc_rgb.weight', 'fc_rgb.bias']