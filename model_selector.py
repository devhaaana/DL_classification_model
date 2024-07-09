from import_module import *
from setting_list import *


def model_selector(args, model="ViT", check_path="./"):
  filename = "checkpoint.pt"
  global deepmodel
    
  if model == "googlenet_2D":
    import models.googlenet.googlenet_2D as deepmodel
    other_options = {"aux_logits" : False}
    
  elif model == "inception_v3_2D":
    import models.inception.inception_v3_2D as deepmodel
    other_options = {"aux_logits" : False,
                     'init_weights' : True}
    
  elif model == "densenet_2D":
    if args.densenet_mode == '121':
      import models.densenet.densenet_2D_121 as deepmodel
    elif args.densenet_mode == '169':
      import models.densenet.densenet_2D_169 as deepmodel
    elif args.densenet_mode == '201':
      import models.densenet.densenet_2D_201 as deepmodel
    elif args.densenet_mode == '264':
      import models.densenet.densenet_2D_264 as deepmodel
    other_options = {}
    
  elif model == "resnet":
    if args.resnet_mode == '18':
      import models.resnet.resnet18_2D as deepmodel
    elif args.resnet_mode == '34':
      import models.resnet.resnet34_2D as deepmodel
    elif args.resnet_mode == '50':
      import models.resnet.resnet50_2D as deepmodel
    elif args.resnet_mode == '101':
      import models.resnet.resnet101_2D as deepmodel
    elif args.resnet_mode == '152':
      import models.resnet.resnet152_2D as deepmodel
    other_options = {}
    
  elif model == "vit_1D":
    if args.vit_mode == 'base':
      import models.vit.vit_1D_base as deepmodel
    elif args.vit_mode == 'large':
      import models.vit.vit_1D_large as deepmodel
    elif args.vit_mode == 'huge':
      import models.vit.vit_1D_huge as deepmodel
    other_options = {'patch_size' : args.patch_size,
                      'dim' : args.dim,
                      'depth' : args.depth,
                      'heads' : args.heads,
                      'mlp_dim' : args.mlp_dim}
    
  elif model == "vit_2D":
    if args.vit_mode == 'base':
      import models.vit.vit_2D_base as deepmodel
    elif args.vit_mode == 'large':
      import models.vit.vit_2D_large as deepmodel
    elif args.vit_mode == 'huge':
      import models.vit.vit_2D_huge as deepmodel
      
    '''
    other_options = {'patch_size' : args.patch_size,
                      'dim' : args.dim,
                      'depth' : args.depth,
                      'heads' : args.heads,
                      'mlp_dim' : args.mlp_dim,
                      'pool' : args.pool,
                      'channels' : args.channels,
                      'dim_head' : args.dim_head,
                      'dropout' : args.dropout,
                      'emb_dropout' : args.emb_dropout}
    '''
    
    if args.dataset == data_list[3] or args.dataset == data_list[4]:
      other_options = {'patch_size' : args.patch_size,
                       'dim' : args.dim,
                       'depth' : args.depth,
                       'heads' : args.heads,
                       'mlp_dim' : args.mlp_dim,
                       'channels' : 1}
    else:
      other_options = {'patch_size' : args.patch_size,
                       'dim' : args.dim,
                       'depth' : args.depth,
                       'heads' : args.heads,
                       'mlp_dim' : args.mlp_dim}
    
  elif model == "vit_3D":
    import models.vit.vit_3D as deepmodel
    other_options = {'patch_size' : args.patch_size,
                       'dim' : args.dim,
                       'depth' : args.depth,
                       'heads' : args.heads,
                       'mlp_dim' : args.mlp_dim}
    
  elif model == "coatnet_2D":
    if args.coatnet_mode == '0':
      import models.coatnet.coatnet0_2D as deepmodel
    elif args.coatnet_mode == '1':
      import models.coatnet.coatnet1_2D as deepmodel
    elif args.coatnet_mode == '2':
      import models.coatnet.coatnet2_2D as deepmodel
    elif args.coatnet_mode == '3':
      import models.coatnet.coatnet3_2D as deepmodel
    elif args.coatnet_mode == '4':
      import models.coatnet.coatnet4_2D as deepmodel
    other_options = {}
    
  '''
  TODO
  CNN 추가
  coatnet7_2D 추가
  
  elif model == "coatnet7_2D":
    import models.coatnet.coatnet7 as deepmodel
    other_options = {}
  '''

  check_fullpath = os.path.join(check_path, filename)

  globals()['other_options'] = other_options
  globals()['check_fullpath'] = check_fullpath
  
  print(f'Model: [{model}]')
  
  return deepmodel, other_options