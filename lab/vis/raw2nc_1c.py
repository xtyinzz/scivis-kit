import os
from netCDF4 import Dataset
import numpy as np

xs = []
zs = []

with open('test/task1_plane.txt', 'r') as f:
  f.readline()
  f.readline()
  line = f.readline()
  while line:
    line = line.strip()
    x,y,z = line.split(',')
    xs.append(float(x))
    zs.append(float(z))
    line = f.readline()

vals = []
with open('test/task1_plane_value.txt', 'r') as f:
  line = f.readline()
  while line:
    line = line.strip()
    vals.append(float(line))
    line = f.readline()


rg = Dataset('sub/task1c.nc', 'w')


rg.createDimension('x', len(xs))
rg.createDimension('y', 1)
rg.createDimension('z', len(zs))

x = rg.createVariable('x', 'f4', ('x',))
y = rg.createVariable('y', 'f4', ('y',))
z = rg.createVariable('z', 'f4', ('z',))
x[:] = xs
y[:] = [0]
z[:] = zs

val = rg.createVariable('value', 'f4', ('x','y','z',))
val[:] = vals

rg.close()