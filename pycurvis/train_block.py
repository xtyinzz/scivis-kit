import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data import SphericalDataset, SphericalBlockDataset, standardization, normalization, train_val
from network.network import MLP
from glob import glob
from util.args import parse_args
from cfg.configs import Config
from util.utils import report_gpumem
import yaml
import signal
from tqdm import tqdm, trange

'''
TODO:
  1. 
'''

def save_checkpoint(model, optim, epoch, batch, loss, log_dir):
  loss_str = f'{float(loss):.2e}'
  pckpt = os.path.join(log_dir, f'ckpt_{epoch:03}_{batch:03}_{loss_str}')
  torch.save({
              'model' : model.state_dict(),
              'optim' : optim.state_dict(),
              'epoch' : epoch,
              'batch' : batch,
              'loss': loss,
              # 'metrics' : metrics.state_dict() if metrics is not None else dict(),
              # 'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
              }, pckpt)
  print(f'checkpoint saved at {pckpt}', flush=True)

def save_checkpoint_block(models, optims, epoch, batch, loss, log_dir):
  loss_str = f'{float(loss):.2e}'
  pckpt = os.path.join(log_dir, f'ckpt_{epoch:03}_{batch:03}_{loss_str}')
  torch.save({
              'models' : [model.state_dict() for model in models],
              'optims' : [optim.state_dict() for optim in optims],
              'epoch' : epoch,
              'batch' : batch,
              'loss': loss,
              # 'metrics' : metrics.state_dict() if metrics is not None else dict(),
              # 'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
              }, pckpt)
  print(f'checkpoint saved at {pckpt}', flush=True)

# return a ckpt containing model, optim, epoch, and loss
def load_checkpoint(ckpt_path, config:Config):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  ckpt = torch.load(ckpt_path, map_location=device)
  # replace state_dict with actual object
  model = config.get_model()
  model.to(device)
  optim = config.get_optim(model)
  model.load_state_dict(ckpt['model'])
  optim.load_state_dict(ckpt['optim'])
  ckpt['model'] = model
  ckpt['optim'] = optim
  return ckpt

def load_checkpoint_block(ckpt_path, config:Config):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  ckpt = torch.load(ckpt_path, map_location=device)
  # replace state_dict with actual object
  num_model = len(ckpt['models'])
  models = [0] * num_model
  optims = [0] * num_model
  for i in trange(num_model):
    models[i] = config.get_model()
    models[i].to(device)
    optims[i] = config.get_optim(models[i])
    models[i].load_state_dict(ckpt['models'][i])
    optims[i].load_state_dict(ckpt['optims'][i])
  ckpt['models'] = models
  ckpt['optims'] = optims
  return ckpt

# load stat_dict for given model and optim
def resume_checkpoint(ckpt_path, model, optim):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  ckpt = torch.load(ckpt_path, map_location=device)
  # replace state_dict with actual object
  model.load_state_dict(ckpt['model'])
  optim.load_state_dict(ckpt['optim'])
  return ckpt['epoch']

# load stat_dict for given models and optims
def resume_checkpoint_block(ckpt_path, models, optims):
  assert len(models) == len(optims)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  ckpt = torch.load(ckpt_path, map_location=device)
  # replace state_dict with actual object
  for i in trange(len(models)):
    models[i].load_state_dict(ckpt['models'][i])
    optims[i].load_state_dict(ckpt['optims'][i])
  return ckpt['epoch']

def main():
  # ************************  signal hanlder ********************************
  # init signal handler to save ckpt prior to exit sbatch
  def handler(signum, frame):
    print('\n\n****** TIME LIMIT APPROACHING, saving model.... ******', signum)
    save_checkpoint_block(models, optims, epoch, ckpt_bi, ckpt_valid_loss/ckpt_bi, log_dir)
  # enable the handler
  signal.signal(signal.SIGUSR1, handler)


  # ************************  main script ********************************
  # parse args
  args = parse_args()
  # get config
  config = Config(args.nncfg)

  # training
  cfg_train = config.get_train_args()
  max_epoch = cfg_train['max_epoch']
  nbatch = cfg_train['batch']
  log_step = cfg_train['log_step']
  batch_log_step = cfg_train['batch_log_step']

  # data
  # dataset_param = config.get_dataset_param()
  # # an item = a batch = a block
  # dataset = SphericalBlockDataset(**dataset_param)
  dataset = config.get_dataset()
  train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

  # val_split = config.config['dataset']['valid_split']
  # train_sampler, val_dataset = train_val(dataset, val_split)

  # train_loader = torch.utils.data.DataLoader(dataset, batch_size=nbatch, 
  #                                           sampler=train_sampler)
  # valid_loader = DataLoader(val_dataset, batch_size=nbatch, shuffle=False)
  

  # model, optim, loss
  log_dir = config.config['model']['log_dir']
  # model_param = config.get_model_param()
  # models = [ MLP(**model_param) for i in range(len(dataset)) ]
  models = [ config.get_model() for i in range(len(dataset)) ]
  optims = [ config.get_optim(model) for model in models ]
  lossl1 = nn.L1Loss()
  lossl2 = nn.MSELoss()
  lossfn = lossl2

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("moving models to device")
  for model in tqdm(models):
    model.to(device)

  # resume training with ckpt={ model:, optim:, epoch:, loss: }if set
  start_epoch = 0
  if args.resume:
    pmodel = os.path.join(log_dir, cfg_train['resume']['ckpt'])
    print("loading checkpoint")
    start_epoch = resume_checkpoint_block(pmodel, models, optims)
    # save_checkpoint_block(models, optims, start_epoch, 1921, 3.02e-1, log_dir)

  print("LR: ", config.config['optim']['param']['lr'])

  # ckpt_bi: number of batches so far, used to check when to save ckpt
  ckpt_bi = 0
  # record average loss across batch steps for ckpt
  ckpt_train_loss = 0
  ckpt_valid_loss = 0
  # if defined batch_log_step is bigger than # of batch, log using # of epoch (i.e. log_step parameter in config)
  if len(train_loader) < batch_log_step:
    batch_log_step = len(train_loader)*log_step
  
  print_bi = 0
  print_train_loss = 0
  print_valid_loss = 0
  print_step = batch_log_step // 10

  for epoch in range(start_epoch, max_epoch):
    print(f"epoch {epoch} started", flush=True)
    # train
    # eloss = 0

    for bi, data in enumerate(train_loader):
      [model.train() for model in models]
      in_data, tar_data = data[0].to(device), data[1].to(device)
      in_data_val, tar_data_val = data[2].to(device), data[3].to(device)

      # forward
      optims[bi].zero_grad()
      pred = models[bi](in_data)
      # with torch.no_grad():
        # print("SHAPES:", pred[:5], tar_data[:5])
      loss = lossfn(pred, tar_data)
      loss.backward()
      optims[bi].step()


      # validate block:
      [model.eval() for model in models]
      with torch.no_grad():
        pred_val = models[bi](in_data_val)
      loss_val = lossfn(pred_val, tar_data_val)
      ckpt_valid_loss += loss_val.item()
      print_valid_loss += loss_val.item()

      # eloss += loss.item()
      ckpt_train_loss += loss.item()
      print_train_loss += loss.item()

      # report intermediate evaluations
      print_bi += 1
      if print_bi % print_step == 0:
        print_train_loss /= print_bi
        print_valid_loss /= print_bi
        print(f'    -BATCH: epoch{epoch:03}-batch{bi:03}-train-loss: {print_train_loss:.4e}   ', end='')
        print(f'valid_loss: {print_valid_loss:.4e}', flush=True)
        print_bi = 0
        print_train_loss = 0
        print_valid_loss = 0
    
      
      # validate and save ckpt
      ckpt_bi += 1
      if ckpt_bi % batch_log_step == 0:
        ckpt_train_loss /= ckpt_bi
        ckpt_valid_loss /= ckpt_bi
        print(f'\n\n+++++++++++++ EPOCH-{epoch:03}-batch{bi:03}-train-loss: {ckpt_train_loss:.4e}  +++++  ', end='')
        print(f'valid loss: {ckpt_valid_loss:.4e} +++++++++++++')
        print('************* SAVING MODEL *************')
        save_checkpoint_block(models, optims, epoch, 0, ckpt_valid_loss, log_dir)
        ckpt_bi = 0
        ckpt_train_loss = 0
        ckpt_valid_loss = 0

      # # train on half of inputs for one epoch
      # if (bi+1)*nbatch >= len(dataset)//2:
      #   break

    # print epoch average loss
    # eloss /= bi+1
    # print(f'\n\n******** EPOCH-{epoch} train loss: {eloss:.4e} ********')

    # validate
    

def validate(model, lossfn, valid_loader):
  device = next(model.parameters()).device
  model.eval()
  vloss = 0
  for bi, data in enumerate(valid_loader):
    in_data, tar_data = data[0].to(device), data[1].to(device)
    # forward
    with torch.no_grad():
      pred = model(in_data)
    loss = lossfn(pred, tar_data)
    vloss += loss.item()
    # print epoch average loss
  vloss /= bi+1
  print(f'valid loss: {vloss:.4e} +++++++++++++')
  return vloss

if __name__ == '__main__':
  main()