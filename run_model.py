from train_module import *
from import_module import *
from model_selector import *
from data_selector import *
from data import *


def run_test(args):
  print("==============================================================================")
  print("\t\t\t[PARAMS SETTING]")
  print('DATE: {} | TIME: {}' .format(args.current_date, args.current_time))
  print('MODEL: {} | COMMENT: {}' .format(args.modelname, args.comment))
  print('DATA: {} | DEVICE: {}' .format(args.dataset, args.device))
  print('EPOCHS: {} | BATCH SIZE: {} | LEARNING RATE: {}' .format(args.epochs, args.batch_size, args.learning_rate))
  print('PATH: {}' .format(args.check_path))

  print("==============================================================================")
  print("\t\t\t[LOAD DATASETS]")
  print("Preparing Training...")
  train_loader, valid_loader, test_loader = data_selector(args)
  print("Loading datasets OK...")

  print("==============================================================================")
  print("\t\t\t[CREATE MODEL]")
  print("Create model...")
  ds = Set_dataset(args)
  input_dim = ds.get_input_dimension()
  target_dim = ds.get_target_dimension()
  
  print(f'input_dim: {input_dim} target_dim: {target_dim}')
  
  deepmodel, other_options = model_selector(args, args.modelname, check_path=args.check_path)
  model = deepmodel.Model(input_dim=input_dim, target_dim=target_dim, **other_options)

  if args.multigpu:
    model = torch.nn.DataParallel(model)
    
  summary(model)
  print("Create model OK...")

  print("==============================================================================")
  print("\t\t\t[MODEL TRAINING]")
  print("Preparing Training...")

  best_val_acc, val_loss, val_acc = 0., 0., 0.
  tr = Model_Trainer(model=model, device=args.device, free_memory=args.free_memory)
  tr.set_parameter(valid_step=args.valid_step, total_epochs=args.epochs)
  tr.set_loaders(train=train_loader, valid=valid_loader, test=valid_loader)
  tr.set_optimizer(opt=args.optimizer, lr=args.learning_rate, weight_decay=args.weight_decay, schedule=args.schedule, scheduler_mode=args.scheduler_mode, **args.scheduler_kwargs)
  tr.set_loss(loss=args.loss_function, reduction=args.loss_reduction)
  tr.set_accuracy(function=args.accuracy_function, reduction=args.accuracy_reduction)
  tr.set_log(log=True, log_print=args.log_print, comment=args.comment, log_folder=args.check_path, checkpoint_step=args.checkpoint_step)
  tr.clip_grad = args.clip_grad

  print("Start Training...")

  for epoch in range(args.epochs):
    loss, acc = next(tr.training)
    
    if len(tr.valid_loss) > 0:
      val_loss, val_acc = tr.last_valid_loss, tr.last_valid_acc
      
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      
  print(f'\nBest acc: {best_val_acc*100:.8f}%')

  print("==============================================================================")
  print("\t\t\t[TESTING MODEL]")
  print("Start Testing...")
  loss, acc = tr.test_model()
  
  print(f'\nTest acc: {acc*100:.8f}%')
  print("==============================================================================")
  print("\t\t\t[SAVING MODEL]")
  torch.save(model, os.path.join(args.check_path, "saved_model.pt"))
  print('Save Model OK...')
  print("==============================================================================")



