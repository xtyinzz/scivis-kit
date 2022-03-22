import gc, sys, psutil, os
from torch import is_tensor
import torch.cuda as tcuda

def report_gpumem(device=0):
  toGb = 1024*1024*1024
  totalm = tcuda.get_device_properties(0).total_memory / toGb
  max_alloc = tcuda.max_memory_allocated() / toGb
  max_rsv = tcuda.max_memory_reserved() / toGb

  palloc = 100*max_alloc / totalm
  prsv = 100*max_rsv / totalm

  print(f'total GPU Mem: {totalm:.4}Gb')
  print(f'max allocated percentage: {palloc:8.4}% --- usage: {max_alloc:.4}Gb')
  print(f'max allocated percentage: {prsv:8.4}% --- usage: {max_rsv:.4}Gb')


def parse_files(fp):
  with open(fp, 'r') as f:
    lines = [line.rstrip() for line in f]
  return lines

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def memReport():
    for obj in gc.get_objects():
        if is_tensor(obj):
            print(type(obj), obj.size())
    
def cpuStats():
    # print(sys.version)
    print(psutil.cpu_percent())
    # print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)