import vtk
from vtkmodules.util import numpy_support
import numpy as np
from tqdm import tqdm

# create a mesh matrix of shape (D1, ..., Di, #ofDim). Di = dimension i length
def get_mesh(*dims):
  mesh_coords = []
  mesh_shape = np.array([len(dim) for dim in dims])
  for i, dim in enumerate(dims):
    # expand shape to everywhere 1 except for the dimension index
    dim_shape = np.ones(len(dims), dtype=int)
    dim_shape[i] = len(dim)
    dim_coords = dim.reshape(dim_shape)

    # repeat the length 1 dimension to match the other dimension lengths
    dim_repeats = mesh_shape.copy()
    dim_repeats[i] = 1
    dim_coords = np.tile(dim_coords, dim_repeats)
    # print(mesh_shape, hex(id(mesh_shape)), hex(id(dim_repeats)))
    mesh_coords.append(dim_coords[..., None])
  
  mesh_coords = np.concatenate(mesh_coords, axis=-1)
  print("meshe generated:", mesh_coords.shape)
  return mesh_coords

def get_vtu(position:np.array, array_dict:dict):
  vtk_position = numpy_support.numpy_to_vtk(position)
  points = vtk.vtkPoints()
  points.SetData(vtk_position)
  data_save = vtk.vtkUnstructuredGrid()
  data_save.SetPoints(points)
  pd = data_save.GetPointData()
  for k, v in array_dict.items():
    vtk_array = numpy_support.numpy_to_vtk(v)
    vtk_array.SetName(k)
    pd.AddArray(vtk_array)
  return data_save

def write_vtu(data_save,filename:str):
  writer = vtk.vtkXMLDataSetWriter()
  writer.SetFileName(filename)
  writer.SetInputData(data_save)
  writer.Write()

def get_vts(dims, points, array_dict={}):
  vtk_grid = vtk.vtkStructuredGrid()
  vtk_grid.SetDimensions(dims)

  # setup grid
  point_coords = numpy_support.numpy_to_vtk(points.reshape(-1, 3))
  vtkPoints = vtk.vtkPoints()
  vtkPoints.SetData(point_coords)
  vtk_grid.SetPoints(vtkPoints)

  # setup values
  pd = vtk_grid.GetPointData()
  for k, v in array_dict.items():
    vtk_array = numpy_support.numpy_to_vtk(v)
    vtk_array.SetName(k)
    pd.AddArray(vtk_array)
  return vtk_grid


# def get_vts(dims, points, array_dict={}):
#   vtk_grid = vtk.vtkStructuredGrid()
#   vtk_grid.SetDimensions(dims)

#   # setup grid
#   point_coords = points.reshape(-1, 3)
#   vtkPoints = vtk.vtkPoints()
#   numpoints = dims[0]*dims[1]*dims[2]
#   vtkPoints.Allocate(numpoints)
#   for i in tqdm(range(numpoints)):
#     vtkPoints.InsertPoint(i, point_coords[i])
#   vtk_grid.SetPoints(vtkPoints)

#   # setup values
#   pd = vtk_grid.GetPointData()
#   for k, v in array_dict.items():
#     vtk_array = numpy_support.numpy_to_vtk(v)
#     vtk_array.SetName(k)
#     pd.AddArray(vtk_array)
#   return vtk_grid

def write_vts(fpath, vts):
  writer = vtk.vtkXMLDataSetWriter()
  writer.SetFileName(fpath)
  writer.SetInputData(vts)
  writer.Write()
