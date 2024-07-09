from import_module import *
from setting_list import *
from run_model import *


def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


def update_args(args):
    args.current_date = datetime.now().strftime('%y%m%d')
    args.current_time = datetime.now().strftime('%H:%M:%S')
    args.start_time = datetime.now()
    
    if torch.backends.mps.is_available():
        args.device = torch.device('mps')
    elif torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
        
    args.best_val_acc = 0.0
    args.save_id, args.save_target, args.save_predict = [], [], []
    
    return args


def make_dir(args):
    args.check_path = f"../trained_models/{args.current_date}_{args.modelname}_{args.dataset}_{args.comment}"
    args.save_path = os.path.dirname(os.path.abspath(__file__)) + "/"
    args.py_save_path = os.path.join(args.check_path, 'code/')
    
    if not os.path.exists(args.check_path):
        os.makedirs(args.check_path, exist_ok=True)
        if args.file_copy:
            os.makedirs(args.py_save_path, exist_ok=True)
    
    return args


def copy_file(args):
    config_py_path = os.path.abspath(__file__)
    copy2(config_py_path, args.py_save_path)


def seed_everything(args):
    if args.use_seed:
        random.seed(args.seed_num)
        os.environ['PYTHONHASHSEED'] = str(args.seed_num)
        np.random.seed(args.seed_num)
        torch.manual_seed(args.seed_num)
        torch.cuda.manual_seed(args.seed_num)
        torch.cuda.manual_seed_all(args.seed_num)
        torch.backends.cudnn.deterministic = True
    
    return args


def duration(args):
    end_time = datetime.now()
    duration_time = end_time - args.start_time
    print("Started at " + str(args.start_time.strftime('%y-%m-%d %H:%M:%S')))
    print("Ended at " + str(end_time.strftime('%y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration_time).split('.')[0])
    
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Trainer')

    parser.add_argument('--comment', type=str, default='test', help='comment')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='STL_10', choices=data_list, help='dataset to use')
    parser.add_argument("--valid_ratio", type=float, default=0.2, help='number of valid ratio')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for data loaders')
    parser.add_argument("--shuffle", type=bool, default=False, help='shuffle')
    
    # Model arguments
    parser.add_argument("--modelname", type=str, default='coatnet_2D', choices=model_list, help='model name')
    parser.add_argument("--coatnet_mode", type=str, default='0', choices=coatnet_size, help='model Size')
    parser.add_argument("--resnet_mode", type=str, default='18', choices=resnet_size, help='model Size')
    parser.add_argument("--densenet_mode", type=str, default='121', choices=densenet_size, help='model Size')
    parser.add_argument("--vit_mode", type=str, default='base', choices=vit_size, help='model Size')
    
    # Learn arguments
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='number of batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='peak learning rate')
    parser.add_argument("--patience", type=int, default=50, help='number of patience')
    parser.add_argument("--valid_step", type=int, default=1, help='number of valid step')
    parser.add_argument("--checkpoint_step", type=int, default=50, help='number of checkpoint step')
    parser.add_argument("--weight_decay", type=int, default=0, help='number of weight decay')
    
    parser.add_argument("--is_training", type=bool, default=True, help='is_training')
    parser.add_argument("--multigpu", type=bool, default=False, help='multi gpu')
    parser.add_argument("--clip_grad", type=bool, default=False, help='clip grad')
    parser.add_argument("--free_memory", type=bool, default=False, help='free memory')
    parser.add_argument("--pin_memory", type=bool, default=True, help='pin memory')

    # Loss, accuracy, optimizer, scheduler arguments
    parser.add_argument("--loss_function", type=str, default='cross_entropy', choices=loss_func_list, help='loss function to use')
    parser.add_argument("--accuracy_function", type=str, default='classifier', choices=acc_func_list, help='accuracy function to use')
    parser.add_argument("--loss_reduction", type=str, default='mean', choices=reduction_list, help='loss reduction to use')
    parser.add_argument("--accuracy_reduction", type=str, default='mean', choices=reduction_list, help='accuracy reduction to use')
    parser.add_argument("--optimizer", type=str, default='Adam', choices=optimizer_list, help='optimizer to use')
    parser.add_argument("--schedule", type=str, default='valid_loss', choices=schedule_list, help='schedule to use')
    parser.add_argument("--scheduler_mode", type=str, default='plateau', choices=scheduler_mode_list, help='scheduler mode to use')
    parser.add_argument("--scheduler_kwargs", type=str, default=scheduler_kwargs, help='scheduler kwargs to use')
    
    # Log arguments
    parser.add_argument("--file_copy", type=bool, default=True, help='file copy')
    parser.add_argument("--log_save", type=bool, default=False, help='log save')
    parser.add_argument("--log_print", type=bool, default=True, help='log print')
    
    # Seed arguments
    parser.add_argument("--use_seed", type=bool, default=False, help='use seed')
    parser.add_argument("--seed_num", type=int, default=1, help='number of seed')
    
    # ViT Arguments
    parser.add_argument("--patch_size", type=int, default=4, help='Patch Size')
    parser.add_argument("--dim", type=int, default=768, help='number of dimension')
    parser.add_argument("--depth", type=int, default=12, help='number of depth')
    parser.add_argument("--heads", type=int, default=12, help='number of heads to use in Multi-head attention')
    parser.add_argument("--mlp_dim", type=int, default=3072, help='number of mlp dimension')
    # parser.add_argument("--pool", type=str, default='cls', help='pool')
    # parser.add_argument("--channels", type=int, default=3, help='number of channels')
    # parser.add_argument("--dim_head", type=int, default=3, help='number of head dimension')
    # parser.add_argument("--dropout", type=float, default=0., help='dropout value')
    # parser.add_argument("--emb_dropout", type=float, default=0., help='embedding dropout value')

    args = parser.parse_args()
    args = update_args(args)
    args = make_dir(args)
    args = seed_everything(args)
    
    # print_args(args)
    if args.file_copy:
        copy_file(args)
        
    run_test(args)

    duration(args)