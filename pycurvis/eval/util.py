import numpy as np
import vtk
from vtk.util import numpy_support
from tqdm import tqdm
import xml.etree.ElementTree as ET

def mae(a:np.ndarray, b:np.ndarray):
  return np.mean(np.abs(a-b))

def mse(a:np.ndarray, b:np.ndarray):
  return np.mean((a-b)*(a-b))

def psnr(a:np.ndarray, gt:np.ndarray):
  vmax = gt.max() - gt.min()
  inner = vmax*vmax / mse(a, gt)
  return 10 * np.log10(inner)

def ssim(a:np.ndarray, b:np.ndarray):
  pass

def write_npy(data_dict, attrs, outpath:str, print_particle=False):

  # # print current particle file infomation
  # if print_particle:
  #   print(tracer[0], '\n')
  #   print(tracer[1], '\n')
  #   num_particle = tracer[0]['ntracers']
  #   for attr,val in tracer[2].items():
  #     print(f'{attr:15}: {str(val.shape):12}', val.dtype)
  #     if val.shape[0] != num_particle:
  #       print('***************Incomplete data***************')
  
  # concat to one matrix and save
  attr_arrs = [data_dict[attr].reshape(-1, 1) for attr in attrs]
  out_npy = np.concatenate(attr_arrs, 1)
  np.save(outpath, out_npy)

'''
Credit: Haoyu Li @OSU GRAVITY lab
'''
def np2vtu(position:np.array, attr_dict:dict={}):
  vtk_position = numpy_support.numpy_to_vtk(position)
  points = vtk.vtkPoints()
  points.SetData(vtk_position)
  data_save = vtk.vtkUnstructuredGrid()
  data_save.SetPoints(points)
  pd = data_save.GetPointData()
  for k, v in attr_dict.items():
    vtk_array = numpy_support.numpy_to_vtk(v)
    vtk_array.SetName(k)
    pd.AddArray(vtk_array)
  return data_save

'''
Credit: Haoyu Li @OSU GRAVITY lab
'''
def write_vtu(position, outpath, attr_dict:dict={}):
  vtu = np2vtu(position, attr_dict)
  writer = vtk.vtkXMLDataSetWriter()
  writer.SetFileName(outpath)
  writer.SetInputData(vtu)
  writer.Write()


def write_pvd(vtkPaths:list, timesteps:list, outPath:str):
  '''
    create a ParaViewData (.pvd, an xml-based file) data containing time-varying tracer data for Pareview.
    Here a reference of PVD file https://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format
    param:
      vtkPath: path of directory containing all vtk files
      timesteps: the timestep each vtk file represents
      outPath: path of output pvd file
  '''
  assert len(vtkPaths) == len(timesteps)

  # root tag
  vtkfile = ET.Element('VTKFile')
  vtkfile.set('type', 'Collection')
  
  # Collection is a child a VKTFile tag containing individual files as children
  collection = ET.SubElement(vtkfile, 'Collection')

  for vtkp, ts in zip(vtkPaths, timesteps):
    dataset = ET.SubElement(collection, 'DataSet')
    dataset.set('timestep', str(ts))
    dataset.set('part', '0')
    dataset.set('file', vtkp)
  
  pvd_et = ET.ElementTree(vtkfile)
  pvd_et.write(outPath, xml_declaration=True)

def np2vts(dims, points:np.ndarray, array_dict:dict={}):
  vtk_grid = vtk.vtkStructuredGrid()
  vtk_grid.SetDimensions(dims)

  # setup grid
  point_coords = points.reshape(-1, 3)
  points = vtk.vtkPoints()
  numpoints = dims[0]*dims[1]*dims[2]
  points.Allocate(numpoints)
  for i in tqdm(range(numpoints)):
    points.InsertPoint(i, point_coords[i])
  vtk_grid.SetPoints(points)

  # setup values
  pd = vtk_grid.GetPointData()
  for k, v in array_dict.items():
    vtk_array = numpy_support.numpy_to_vtk(v)
    vtk_array.SetName(k)
    pd.AddArray(vtk_array)
  
  return vtk_grid

def write_vts(fpath, vts):
  writer = vtk.vtkXMLDataSetWriter()
  writer.SetFileName(fpath)
  writer.SetInputData(vts)
  writer.Write()
    