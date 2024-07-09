data_list = ['CIFAR_10', 'CIFAR_100', 'ImageNet', 'MNIST', 'Fashion_MNIST', 'SVHN', 'STL_10']

model_list = ['vanilla',
            'cnn',
            'googlenet_1D',
            'googlenet_2D',
            'inception_v3_1D',
            'inception_v3_2D',
            'densenet_1D',
            'densenet_2D',
            'resnet',
            'vit_1D',
            'vit_2D',
            'vit_3D',
            'coatnet_2D']
coatnet_size = ['0', '1', '2', '3', '4']
densenet_size = ['121', '169', '201', '264']
resnet_size = ['18', '34', '50', '101', '152']
vit_size = ['base', 'large', 'huge']

loss_func_list = ['cross_entropy', 'binary_cross_entropy', 'mse_loss', 'l1_loss', 'nll_loss', 'smooth_l1_loss']
acc_func_list = ['classifier']
reduction_list = ['mean']
optimizer_list = ['Adam', 'SGD', 'Adagrad']
schedule_list = ['train_loss', 'train_acc', 'valid_loss', 'valid_acc']
scheduler_mode_list = ['plateau',
                        'lambda',
                        'multiplicative',
                        'step',
                        'multistep',
                        'exponential',
                        'cosine',
                        'cyclic',
                        'onecycle',
                        'cosineAnnealingWarmRestarts',
                        'custom']
scheduler_kwargs = {
                    'factor'  : 0.5,
                    'patience': 10,
                    # 'verbose' : True,
                }