import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, dataloader
from tqdm import tqdm
from cfg.configs import Config
from util.args import parse_args
from data.data import SphericalDataset, train_val
from train import load_checkpoint
from eval.util import mae, mse, psnr, np2vts, write_vts
from vis_io import get_vts, write_vts

def get_data(cfgObj, **dataset_param):
  dataset = SphericalDataset(**dataset_param)
  return dataset

def get_model(model_path, nncfg):
  ckpt = load_checkpoint(model_path, nncfg)
  model = ckpt['model']
  return model

def inference(model, dataset, dataloader):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  # inputs = []
  preds = []
  gts = []
  with torch.no_grad():
    for bi, data in enumerate(tqdm(dataloader)):
      in_data = data[0].to(device)
      pred = model(in_data)

      # accumulate complete data
      # inputs.append(pos_time.detach().cpu())
      preds.append(pred.detach().cpu())
      gts.append(data[1])

      # if (bi+1)*dataloader.batch_size >= len(dataset)//4:
      #   break
  
  # inputs = torch.concat(inputs, 0)
  preds = torch.concat(preds, 0)
  gts = torch.concat(gts, 0)

  print(f'\nSHAPES: ', preds.shape, gts.shape ,'\n')

  return preds.numpy(), gts.numpy()

def evaluate(pred:np.ndarray, gt:np.ndarray, verbose=True):
  '''
  evaluate on MAE, MSE and PSNR
  '''

  vmae = mae(pred, gt)
  vmse = mse(pred, gt)
  # vpsnr = psnr(pred, gt)

  if verbose:
    print(f' MAE: {vmae}')
    print(f' MSE: {vmse}')
    # print(f'PSNR: {vpsnr}')

  return vmae, vmse

def main():
  args = parse_args()
  cfgObj = Config(args.testcfg)
  nncfg = Config(args.nncfg)
  testcfg = cfgObj.config
  
  # get model
  model_path = testcfg['model_path']
  model = get_model(model_path, nncfg)

  # get data

  dataset = get_data(cfgObj, **nncfg.get_dataset_param())

  nbatch = nncfg.config['train']['batch']
  dataloader = DataLoader(dataset, batch_size=nbatch, shuffle=False)

  # train_sampler, val_dataset = train_val(dataset, .2)

  # dataloader = torch.utils.data.DataLoader(dataset, batch_size=nbatch, 
  #                                           sampler=train_sampler)

  # valid_loader = DataLoader(val_dataset, batch_size=nbatch, shuffle=False)



  pred, gt = inference(model, dataset, dataloader)

  _, _ = evaluate(pred, gt, verbose=True)

  # write data
  out_dir = testcfg['out_dir']
  vis_out_dir = testcfg['vis_out_dir']
  out_name = testcfg['out_name']

  pred = dataset.outpp.inverse_transform(pred)
  gt = dataset.outpp.inverse_transform(gt)


  # output numpy
  out_path = os.path.join(out_dir, f"pred_{out_name}.npy")
  np.save(out_path, pred)
  out_path = os.path.join(out_dir, f"gt_{out_name}.npy")
  np.save(out_path, gt)


  pred_vts = get_vts(dataset.dims, pred)
  gt_vts = get_vts(dataset.dims, gt)
  # output vts for visualzation
  out_path = os.path.join(out_dir, f"pred_{out_name}.vts")
  write_vts(out_path, pred_vts)
  out_path = os.path.join(out_dir, f"gt_{out_name}.vts")
  write_vts(out_path, gt_vts)



if __name__ == '__main__':
  main()