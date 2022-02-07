import os
from netCDF4 import Dataset
import numpy as np

rg = Dataset('sub/task1c.nc', 'w')

dmin, dmax = [-1, 1]
step = 2 / 299
dim_points = np.arange(dmin, dmax + step, step)

rg.createDimension('x', dim_points.shape[0])
rg.createDimension('y', dim_points.shape[0])
rg.createDimension('z', dim_points.shape[0])

x = rg.createVariable('x', 'f4', ('x',))
y = rg.createVariable('y', 'f4', ('y',))
z = rg.createVariable('z', 'f4', ('z',))
x[:] = dim_points
y[:] = dim_points
z[:] = dim_points

func_val = np.fromfile('data/task1a.raw', dtype=np.float32)
func_val = func_val.reshape(x.shape[0], y.shape[0], z.shape[0])
val = rg.createVariable('value', 'f4', ('x','y','z',))
val[:] = func_val

rg.close()