from data import *
from setting_list import *


def data_selector(args):
    if args.dataset == data_list[0] or args.dataset == data_list[1]:
        data = CIFAR_dataset(args)
    elif args.dataset == data_list[2]:
        data = ImageNet_data(args)
    elif args.dataset == data_list[3] or args.dataset == data_list[4]:
        data = MNIST_data(args)
    elif args.dataset == data_list[5]:
        data = SVHN_data(args)
    elif args.dataset == data_list[6]:
        data = STL_data(args)
        
    train_loader, valid_loader, test_loader = data.load_data()
    
    print(f'Data: [{args.dataset}]')
    print('Train: ', len(train_loader.dataset))
    print('Valid: ', len(valid_loader.dataset))
    print('Test: ', len(test_loader.dataset))
    
    return train_loader, valid_loader, test_loader