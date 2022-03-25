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

def get_vtu(position:np.array, scalar_fields={}, vector_fields={}):
  vtk_position = numpy_support.numpy_to_vtk(position)
  points = vtk.vtkPoints()
  points.SetData(vtk_position)
  data_save = vtk.vtkUnstructuredGrid()
  data_save.SetPoints(points)

  # setup values
  pd = data_save.GetPointData()
  for i, (k, v) in enumerate(scalar_fields.items()):
    vtk_array = numpy_support.numpy_to_vtk(v)
    vtk_array.SetName(k)
    if i == 0:
      pd.SetScalars(vtk_array)
    else:
      pd.AddArray(vtk_array)
  
  for i, (k, v) in enumerate(vector_fields.items()):
    vtk_array = numpy_support.numpy_to_vtk(v)
    vtk_array.SetName(k)
    if i == 0:
      pd.SetVectors(vtk_array)
    else:
      pd.AddArray(vtk_array)
  
  return data_save

def write_vtu(fpath:str, vtu):
  writer = vtk.vtkXMLDataSetWriter()
  writer.SetFileName(fpath)
  writer.SetInputData(vtu)
  writer.Write()

def get_vts(dims, points, scalar_fields={}, vector_fields={}):
  vtk_grid = vtk.vtkStructuredGrid()
  vtk_grid.SetDimensions(dims)

  # setup grid
  point_coords = numpy_support.numpy_to_vtk(points.reshape(-1, 3))
  vtkPoints = vtk.vtkPoints()
  vtkPoints.SetData(point_coords)
  vtk_grid.SetPoints(vtkPoints)

  # setup values
  pd = vtk_grid.GetPointData()
  for i, (k, v) in enumerate(scalar_fields.items()):
    vtk_array = numpy_support.numpy_to_vtk(v)
    vtk_array.SetName(k)
    if i == 0:
      pd.SetScalars(vtk_array)
    else:
      pd.AddArray(vtk_array)
  
  for i, (k, v) in enumerate(vector_fields.items()):
    vtk_array = numpy_support.numpy_to_vtk(v)
    vtk_array.SetName(k)
    if i == 0:
      pd.SetVectors(vtk_array)
    else:
      pd.AddArray(vtk_array)
  
  return vtk_grid

def write_vts(fpath, vts):
  writer = vtk.vtkXMLStructuredGridWriter()
  writer.SetFileName(fpath)
  writer.SetInputData(vts)
  writer.Write()


# x,y,z coordiantes either numpy / vtk array 
def get_vtr(dims, xCoords, yCoords, zCoords, scalar_fields={}, vector_fields={}):
  assert type(xCoords) == type(yCoords) and type(yCoords) == type(zCoords)
  assert isinstance(xCoords, np.ndarray) or isinstance(xCoords, vtk.vtkDataArray)

  grid = vtk.vtkRectilinearGrid()
  grid.SetDimensions(dims)

  if isinstance(xCoords, np.ndarray):
    xCoords = numpy_support.numpy_to_vtk(xCoords)
    yCoords = numpy_support.numpy_to_vtk(yCoords)
    zCoords = numpy_support.numpy_to_vtk(zCoords)

  grid.SetXCoordinates(xCoords)
  grid.SetYCoordinates(yCoords)
  grid.SetZCoordinates(zCoords)

  pd = grid.GetPointData()
  for i, (k, v) in enumerate(scalar_fields.items()):
    vtk_array = numpy_support.numpy_to_vtk(v)
    vtk_array.SetName(k)
    if i == 0:
      pd.SetScalars(vtk_array)
    else:
      pd.AddArray(vtk_array)
  
  for i, (k, v) in enumerate(vector_fields.items()):
    vtk_array = numpy_support.numpy_to_vtk(v)
    vtk_array.SetName(k)
    if i == 0:
      pd.SetVectors(vtk_array)
    else:
      pd.AddArray(vtk_array)

  return grid

def write_vtr(fpath, vtr):
  writer = vtk.vtkXMLRectilinearGridWriter()
  writer.SetFileName(fpath)
  writer.SetInputData(vtr)
  writer.Write()

# return vtkFloatArray holding data like np.arange(args)
def vtk_arange(*args):
  arr = vtk.vtkFloatArray()
  coords = np.arange(*args)
  arr.Allocate(len(coords))
  for i in coords:
    arr.InsertNextValue(i)
  return arr

# return vtkFloatArray holding data like np.linspace(args)
def vtk_linspace(*args):
  arr = vtk.vtkFloatArray()
  coords = np.linspace(*args)
  arr.Allocate(len(coords))
  for i in coords:
    arr.InsertNextValue(i)
  return arr