import sklearn
import torch
import numpy as np
from torch.utils.data import Dataset, SubsetRandomSampler, Subset
from util.utils import cpuStats
from os import path
from tqdm import tqdm, trange
from util.utils import report_gpumem, cpuStats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

def minmax_scale(x, new_min=-1., new_max=1.):
  x_max, x_min = [x.max(1, keepdim=True)[0], x.min(1, keepdim=True)[0]]
  return (x - x_min) / (x_max - x_min) * (new_max - new_min) + new_min

def parse_filenames(pfiles):
  '''
  Given path to a txt file containing data file paths, whitespace seperated,
  Return file names as a list.
  '''
  with open(pfiles, 'r') as f:
    files = [line.rstrip() for line in f]
  return files

def write_stats(particle, outpath):
  '''
  get min max value of particle data (N, C) and write to npy file

  return: (S, C), S={min, max}
  '''
  attrmin = particle.min(0, keepdims=True)
  attrmax = particle.max(0, keepdims=True)
  stats = np.concatenate([attrmin, attrmax], 0)
  np.save(outpath, stats)
  print(f'stats shaped {stats.shape} saved {outpath}')


def standardization(x:torch.Tensor, dim=None):
  if dim is None:
    xmean = x.mean()
    xstd = x.std()
  else:
    xmean = torch.mean(x, dim=dim, keepdim=True)[0]
    xstd = torch.min(x, dim=dim, keepdim=True)[0]
  return (x - xmean / xstd)

def normalization(x:np.ndarray, new_min=-1, new_max=1, dim=None):
  if dim is None:
    curr_max = x.max()
    curr_min = x.min()
  else:
    curr_max = np.max(x, dim, keepdims=True)
    curr_min = np.min(x, dim, keepdims=True)
  return (x - curr_min) / (curr_max - curr_min) * (new_max - new_min) + new_min

# def normalization(x:torch.Tensor, new_min=-1, new_max=1, dim=None):
#   if dim is None:
#     curr_max = x.max()
#     curr_min = x.min()
#   else:
#     curr_max = torch.max(x, dim=dim, keepdim=True)[0]
#     curr_min = torch.min(x, dim=dim, keepdim=True)[0]
#   return (x - curr_min) / (curr_max - curr_min) * (new_max - new_min) + new_min

def stand_norm(x):
  x = standardization(x)
  x = normalization(x)
  return x

def train_val_indices(dataset_size, valid_split):
  random_seed= 42
  # Creating data indices for training and validation splits:
  indices = np.arange(dataset_size)
  np.random.seed(random_seed)
  np.random.shuffle(indices)
  val_count = int(np.floor(valid_split * dataset_size))
  train_indices, val_indices = indices[val_count:], indices[:val_count]

  return train_indices, val_indices

def train_val(dataset: Dataset, valid_split):
  validation_split = .2
  shuffle_dataset = True
  random_seed= 42

  # Creating data indices for training and validation splits:
  dataset_size = len(dataset)
  train_indices, val_indices = train_val_indices(dataset_size, valid_split)

  # Creating training sampler and validation dataset:
  train_sampler = SubsetRandomSampler(train_indices)
  val_dataset = Subset(dataset, val_indices)
  
  return train_sampler, val_dataset

  train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
  validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  sampler=valid_sampler)
  validation_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)



class NormScaler():
  def __init__(self, x, new_range=[-1,1], dim=None):
    if dim is None:
      curr_max = x.max()
      curr_min = x.min()
    else:
      curr_max = torch.max(x, dim=dim, keepdim=True)[0]
      curr_min = torch.min(x, dim=dim, keepdim=True)[0]

    self.range = [curr_min, curr_max]
    self.new_range = new_range

  def __call__(self, x):
    return (x - self.range[0]) / (self.range[1] - self.range[0]) * (self.new_range[1] - self.new_range[0]) + self.new_range[0]
  
  # swap new and old minmax (e.g. for use of recovering orignal range)
  def swap_minmax(self):
    tmp = self.range
    self.range = self.new_range
    self.new_range = tmp

  # return the orignal data from tranformed data
  def recover(self, transed_x):
    self.swap_minmax()
    x = self(transed_x)
    self.swap_minmax()
    return x

class StandScaler():
  def __init__(self, x, dim=None):
    if dim is None:
      mean = x.mean()
      std = x.std()
    else:
      mean = torch.mean(x, dim=dim, keepdim=True)[0]
      std = torch.std(x, dim=dim, keepdim=True)[0]

    self.mean = mean
    self.std = std

  def __call__(self, x):
    return (x - self.mean / self.std)
  
  # return the orignal data from tranformed data
  def recover(self, transed_x):
    return transed_x * self.std + self.mean

# class TransPipeline():
#   def __init__(self):
#     self.norm = NormScaler()
#     self.stand = StandScaler()

def fitTransPipeline(x: np.ndarray):
  pp = Pipeline1D([
    ('stand', StandardScaler()),
    ('norm', MinMaxScaler((-1, 1))),
  ])
  x_shape = x.shape
  x = pp.fit_transform(x)
  return x, pp

# global pipeline (norm and stand coordinates in SphericalDataset's case)
class Pipeline1D(Pipeline):
  def fit_transform(self, X, y=None, **fit_params):
    x_shape = X.shape
    X = X.reshape(-1, 1)
    X = super().fit_transform(X, y, **fit_params)
    return X.reshape(x_shape)

  def fit(self, X, y=None, **fit_params):
    x_shape = X.shape
    X = X.reshape(-1, 1)
    return super()._fit(X, y, **fit_params)
  
  def transform(self, X):
    x_shape = X.shape
    X = X.reshape(-1, 1)
    X = super()._transform(X)
    return X.reshape(x_shape)

  def inverse_transform(self, X):
    x_shape = X.shape
    X = X.reshape(-1, 1)
    X = super()._inverse_transform(X)
    return X.reshape(x_shape)

'''
    data_dir: "../data/",
    curv_idx: [0,1,2],
    cart_idx: [3,4,5],
'''
class SphericalDataset(Dataset):
  def __init__(self, data_path, curv_idx, cart_idx, intrans=fitTransPipeline, outtrans=fitTransPipeline):
    self.data_path = data_path
    self.coords = np.load(data_path)
    self.dims = self.coords.shape[:-1]
    self.curv_idx = curv_idx
    self.cart_idx = cart_idx
    self.curv = self.coords[..., curv_idx]
    self.cart = self.coords[..., cart_idx]
    assert len(self.curv) == len(self.cart)

    if intrans is not None:
      self.cart_prep, self.inpp = fitTransPipeline(self.cart.reshape(-1, len(cart_idx)))
      # print(self.cart.mean())
    if outtrans is not None:
      self.curv_prep, self.outpp = fitTransPipeline(self.curv.reshape(-1, len(curv_idx)))
      # print(self.curv.mean())
  
    self.cart_prep = torch.tensor(self.cart_prep)
    self.curv_prep = torch.tensor(self.curv_prep)

  def __len__(self):
    return len(self.curv_prep)

  def __getitem__(self, idx):
    return self.cart_prep[idx], self.curv_prep[idx]

class SphericalBlockDataset(Dataset):
  def __init__(self, data_path, curv_idx, cart_idx, intrans=fitTransPipeline, outtrans=fitTransPipeline, downscale=8, axis=[0,1,2], valid_split=0.2):
    self.data_path = data_path
    self.coords = np.load(data_path)
    self.dims = self.coords.shape[:-1]
    self.curv_idx = curv_idx
    self.cart_idx = cart_idx
    self.curv = self.coords[..., curv_idx]
    self.cart = self.coords[..., cart_idx]
    assert self.curv.shape == self.cart.shape

    self.downscale = downscale
    self.axis = axis
    self.curv = torch.tensor(self.curv)
    self.cart = torch.tensor(self.cart)
    self.carts = self.get_block(self.cart, self.downscale, self.axis)
    self.curvs = self.get_block(self.curv, self.downscale, self.axis)
    # self.cart = self.coords[..., cart_idx].reshape(-1, len(cart_idx))
    
    self.hasIntrans = False
    self.hasOutrans = False
    if intrans is not None:
      self.hasIntrans = True
      print("transforming inputs")
      for i, coords in enumerate(self.carts):
        new_coord, pp  = fitTransPipeline(coords.numpy())
        self.carts[i] = (new_coord, pp)
      # print(self.cart.mean())

    if outtrans is not None:
      self.hasOutrans = True
      print("transforming targets")
      for i, coords in enumerate(self.curvs):
        new_coord, pp  = fitTransPipeline(coords.numpy())
        self.curvs[i] = (new_coord, pp)
      # print(self.curv.mean())

    # train valid split for cart
    print("creating train test splits for inputs")
    self.cart_train, self.cart_valid = self.train_valid(self.carts, valid_split)
    print("creating train test splits for targets")
    self.curv_train, self.curv_valid = self.train_valid(self.curvs, valid_split)
  
  def train_valid(self, x, valid_split):
    x_train = [0] * len(x)
    x_valid = [0] * len(x)
    for i, coords in enumerate(x):
      if isinstance(coords, tuple):
        coords = coords[0]
      dataset_size = len(coords)
      train_indices, val_indices = train_val_indices(dataset_size, valid_split)
      x_train[i] = coords[train_indices]
      x_valid[i] = coords[val_indices]
    return x_train, x_valid

  def get_block(self, coords, downscale, axis):
    dim_chunks = [np.arange(0, length+1, length//downscale) for length in coords.shape[:-1]]
    coords_block = []
    dimi, dimj, dimk = dim_chunks
    for i in range(len(dimi)-1):
      block_i = [dimi[i], dimi[i+1]]
      for j in range(len(dimj)-1):
        block_j = [dimj[j], dimj[j+1]]
        for k in range(len(dimk)-1):
          block_k = [dimk[k], dimk[k+1]]
          # print(block_i, block_j, block_k)
          block = coords[ block_i[0]:block_i[1], block_j[0]:block_j[1], block_k[0]:block_k[1] ]
          # reshape coord block to (N, 3)
          coords_block.append(block.reshape(-1, block.shape[-1]))
    return coords_block

  def __len__(self):
    return len(self.cart_train)

  def __getitem__(self, idx):
    return (
      self.cart_train[idx], self.curv_train[idx],
      self.cart_valid[idx], self.curv_valid[idx],
    )

# trajectory data: (N, Time, C), C=6: harm, cart
# input: (pos_t, t)
# target: (pos_t+1 - pos_t)
class TrajectoryDeltaDataset(Dataset):
  def __init__(self, data_dir, file_name, len_file_name, in_idx, input_trans=None):
    self.fp = path.join(data_dir, file_name)
    self.len_fp = path.join(data_dir, len_file_name)
    self.input_trans = input_trans
    self.in_idx = in_idx
    print("loading")
    data = np.load(self.fp)
    self.timelen = np.load(self.len_fp)
    self.input = torch.tensor(data[:, :, in_idx], dtype=torch.float32)
    # self.input_coord = torch.tensor(data[:, 0, in_idx], dtype=torch.float32)

    if self.input_trans is not None:
      self.input = self.input_trans(self.input, dim=None)
      
    input_list = []
    target_list = []
    
    # PREDICT DELTA
    # for i in trange(self.timelen.shape[0]):
    #   curr_pos = self.input[i, :(self.timelen[i]-1)]
    #   curr_time = torch.arange(0, self.timelen[i]-1, 1).unsqueeze(-1)
    #   next_pos = self.input[i, 1:self.timelen[i]]
    #   delta = next_pos - curr_pos

    #   input_list.append(torch.concat([curr_pos, curr_time], dim=-1))
    #   target_list.append(delta)
    # print(len(input_list))
    # self.input = torch.concat(input_list, dim=0)
    # self.target = torch.concat(target_list, dim=0)

    # PREDICT FLOW MAP

    self.starts = self.input[:, 0, :].view(-1, len(in_idx))

    # for each pathline i
    for i in trange(len(self.timelen)):
      curr_start = self.starts[i].unsqueeze(0).repeat(self.timelen[i]-1, 1)
      curr_time = torch.arange(1, self.timelen[i], 1).unsqueeze(-1)
      curr_target = self.input[i, 1:self.timelen[i], :]

      input_list.append(torch.concat([curr_start, curr_time], dim=-1))
      target_list.append(curr_target)

    self.input = torch.concat(input_list, dim=0)
    self.target = torch.concat(target_list, dim=0)

    
    print("Data processed.")
    print("input shape:", self.input.shape)
    print("target shape:", self.target.shape)


  def __len__(self):
    return sum(self.timelen) - len(self.timelen)

  def __getitem__(self, idx):
    # random sample
    return self.input[idx], self.target[idx]

class VBHGridDataset(Dataset):
  def __init__(self, data_dir, files_path, in_idx, target_idx, input_trans=standardization, target_trans=None):
    self.dir = data_dir
    self.in_idx = in_idx
    self.target_idx = target_idx
    self.files = parse_filenames(files_path)
    self.input_trans = input_trans
    self.target_trans = target_trans

    self.pdata = path.join(self.dir, self.files[0])
    data = np.load(self.pdata)
    self.grid_dim = data.shape[:-1]

    data = torch.tensor(data, dtype=torch.float32)
    data = data.reshape(-1, data.shape[-1])

    self.coord = data[:, in_idx]
    self.val = data[:, target_idx]

    if self.input_trans is not None:
      self.coord = self.input_trans(self.coord, dim=0)
    
    if self.target_trans is not None:
      self.val = self.target_trans(self.val)

  def __len__(self):
    return len(self.coord)

  def __getitem__(self, idx):
    return self.coord[idx], self.val[idx]

# tracer features:
# ['Xcart/0-2', 'mass/3', 'rho/4', 'T/5', 'Press/6', 'uu/7', 'Ye/8', 'Ye_em/9']
# gt_path: full res tracer
# sampple_path: downsampeld tracer
class TracerDataset(Dataset):
  def __init__(self, gt_dir, sample_dir, files_path, attr_idx=[0,1,2,8], downscale=2, transform=None, target_transform=minmax_scale,):
    # assert len(gt_path) == len(sample_path)

    self.gt_dir = gt_dir
    self.smp_dir = sample_dir
    self.files = parse_filenames(files_path)
    self.attr_idx = attr_idx
    self.transform = transform
    self.downscale = downscale
    self.target_transform = target_transform

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    gt_path = path.join(self.gt_dir, self.files[idx])
    smp_path = path.join(self.smp_dir, self.files[idx])
    tracer_target = torch.tensor(np.load(gt_path))[:,self.attr_idx]
    downsampled = torch.tensor(np.load(smp_path))[:,self.attr_idx]
    if self.transform:
        downsampled = self.transform(downsampled)
    if self.target_transform:
        tracer_target = self.target_transform(tracer_target)
    return downsampled, tracer_target

'''
ABLATION4: input scaling
1. scale each ts indivudally
  1.1: downsample and GT same scale
  1.2: downsample and GT individually scale
2. scale each ts with global min max
  2.1: downsample and GT same scale
  2.2: downsample and GT individually scale

'''

# # minmax scale and produce downsampled.
# for idx in range(len(ftracers)):
#   ftracer = ftracers[idx]
#   tracer = torch.tensor(np.load(ftracer))
#   tracer = minmax_scale(tracer)

#   num_samples = tracer.size(0) // 2
#   shuffled = tracer[torch.randperm(tracer.size(0)), :]
#   downsampled = shuffled[:num_samples]
#   np.save(f'{smp_dir}/{ftracer[-18:]}', downsampled)
#   np.save(ftracer, tracer)