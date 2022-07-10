import numpy as np

def deg2rad(deg):
  return deg * np.pi / 180

# for numerical isse
def mysin(a):
  val = np.sin(a)
  val = close_round(val, 0)
  val = close_round(val, 1)
  return val

def mycos(a):
  val = np.cos(a)
  val = close_round(val, 0)
  val = close_round(val, 1)
  return val

def close_round(val, test_val, abs_bounds=1e-12):
  isclose = np.abs(test_val - val) < abs_bounds
  # print(isclose)
  if isinstance(val, float) or isinstance(val, int) or np.float32:
    val_cp = test_val if isclose else val
  else:
    val_cp = val.copy()
    val_cp[isclose] = test_val
  return val_cp

def sph2car(r, theta, phi):
  # x_coef = np.sin(phi)*np.cos(theta)
  # y_coef = np.around(np.sin(phi)*np.sin(theta), decimals=10)
  # z_coef = np.around(np.cos(phi), decimals=10)
  x = r*mysin(phi)*mycos(theta)
  y = r*mysin(phi)*mysin(theta)
  z = r*mycos(phi)
  
  return np.array([x,y,z])

def car2sph(x, y, z):
  # assert (x or y) != 0
  r = np.sqrt(x*x + y*y + z*z)
  theta = np.arctan(y/x)
  phi = np.arctan(np.sqrt(x*x + y*y) / z)
  # if x > 0:
  #   phi = np.arctan(y/x)
  # elif x < 0 and y >= 0:
  #   phi = np.arctan(y/x)+np.pi
  # elif x < 0 and y < 0:
  #   phi == np.arctan(y/x)-np.pi
  # elif x == 0 and y > 0:
  #   phi = np.pi/2
  # elif x == 0 and y < 0:
  #   phi = -np.pi/2

  return np.array([r,theta,phi])
# //
# //		    6________7  high-vtx
# //		   /|       /|
# //		  / |      / |
# //		4/_______5/  |
# //		|  2|___ |___|3
# //		|  /     |  /
# //		| /      | /
# //		|/_______|/
# //		0        1
# //  low_vtx

# //
# //		 011_________111  high-vtx
# //		   /|       /|
# //		  / |      / |
# //	001/_____101/  |
# //		|010|___ |___|110
# //		|  /     |  /
# //		| /      | /
# //		|/_______|/
# //	000       100

# //
# //		    8________7  high-vtx
# //		   /|       /|
# //		  / |      / |
# //		5/_______6/  |
# //		|  4|___ |___|3
# //		|  /     |  /
# //		| /      | /
# //		|/_______|/
# //		1        2
# //  low_vtx

def trilerp(coord: np.ndarray, low_vtx: np.ndarray, high_vtx: np.ndarray, cell_val: np.ndarray):
  # x0,y0,z0 = low_vtx
  # x1,y1,z1 = high_vtx
  # x, y, z = coord

  # low_vtx, high_vtx, coord = normalization(np.array([low_vtx, high_vtx, coord]), new_min=0, dim=0)

  # TODO: Why trilerp doesn't work if low and high are not normalized to 1
  # print(coord, low_vtx, high_vtx)
  # print(coord)
  x, y, z = (coord - low_vtx) / (high_vtx - low_vtx)
  # print(x,y,z)
  # low_vtx = np.zeros(3)
  # high_vtx = np.ones(3)
  # x0,y0,z0 = low_vtx
  # x1,y1,z1 = high_vtx

  v000 = cell_val[0,0,0]
  v100 = cell_val[1,0,0]
  v010 = cell_val[0,1,0]
  v110 = cell_val[1,1,0]
  v001 = cell_val[0,0,1]
  v101 = cell_val[1,0,1]
  v011 = cell_val[0,1,1]
  v111 = cell_val[1,1,1]
  # print(x, y, z)
  # print(x0, y0, z0)
  # print(x1, y1, z1)
  # print((x0-x1)*(y0-y1)*(z0-z1))
  # print(((x0-x1)*(y0-y1)*(z0-z1)))
  # for cord in [v000, v100, v010, v110, v001, v101, v011, v111]:
  #   print(cord)
  
  # a0 = (
  #   (-v000*x1*y1*z1 + v001*x1*y1*z0 + v010*x1*y0*z1 - v011*x1*y0*z0 + 
  #   v100*x0*y1*z1 - v101*x0*y1*z0 - v110*x0*y0*z1 + v111*x0*y0*z0)
  # )/ ((x0-x1)*(y0-y1)*(z0-z1))
  # a1 = (
  #   (v000*z1*y1 - v001*z0*y1 - v010*z1*y0 + v011*z0*y0 - 
  #   v100*z1*y1 + v101*z0*y1 + v110*z1*y0 - v111*z0*y0)
  # )/ ((x0-x1)*(y0-y1)*(z0-z1))
  # a2 = (
  #   (v000*x1*z1 - v001*x1*z0 - v010*x1*z1 + v011*x1*z0 - 
  #   v100*x0*z1 + v101*x0*z0 + v110*x0*z1 - v111*x0*z0)
  # )/ ((x0-x1)*(y0-y1)*(z0-z1))
  # a3 = (
  #   (v000*x1*y1 - v001*x1*y1 - v010*x1*y0 + v011*x1*y0 - 
  #   v100*x0*y1 + v101*x0*y1 + v110*x0*y0 - v111*x0*y0)
  # )/ ((x0-x1)*(y0-y1)*(z0-z1))
  # a4 = (
  #   (-v000*z1 + v001*z0 + v010*z1 - v011*z0 + 
  #   v100*z1 - v101*z0 - v110*z1 + v111*z0)
  # )/ ((x0-x1)*(y0-y1)*(z0-z1))
  # a5 = (
  #   (-v000*y1 + v001*y1 + v010*y0 - v011*y0 + 
  #   v100*y1 - v101*y1 - v110*y0 + v111*y0)
  # )/ ((x0-x1)*(y0-y1)*(z0-z1))
  # a6 = (
  #   (-v000*x1 + v001*x1 + v010*x1 - v011*x1 + 
  #   v100*x0 - v101*x0 - v110*x0 + v111*x0)
  # )/ ((x0-x1)*(y0-y1)*(z0-z1))
  # a7 = (
  #   (v000 - v001 - v010 + v011 -v100 + v101 + v110 - v111)
  # )/ ((x0-x1)*(y0-y1)*(z0-z1))
  
  
  # //
  # //		    8________7  high-vtx
  # //		   /|       /|
  # //		  / |      / |
  # //		5/_______6/  |
  # //		|  4|___ |___|3
  # //		|  /     |  /
  # //		| /      | /
  # //		|/_______|/
  # //		1        2
  # //  low_vtx

  #   [v000, v100, v110, v010,
  #  v001, v101, v111, v011,]
  a0 = v000
  a1 = -v000 + v100
  a2 = -v000 + v010
  a3 = -v000 + v001
  a4 = v000 - v100  + v110 - v010
  a5 = v000 - v100 - v001 + v101
  a6 = v000 - v010 - v001 + v011
  a7 = -v000 +v100 -v110 +v010 +v001 -v101 +v111 -v011

  interpolant = (a0 + a1*x + a2*y + a3*z + a4*x*y + a5*x*z + a6*y*z + a7*x*y*z)
  coeff = np.array([a0, a1, a2, a3, a4, a5, a6, a7]).T
  # coeff1 = np.array([aa0, aa1, aa2, aa3, aa4, aa5, aa6, aa7]).T
  # print(coeff == coeff1)
  # print(coeff)
  # print(coeff1)
  return interpolant, coeff

# G
def comp2phys(comp: np.ndarray, vtx_low, vtx_high, cell_coords):
  phys_est_x, a = trilerp(comp, vtx_low, vtx_high, cell_coords[..., 0])
  phys_est_y, b = trilerp(comp, vtx_low, vtx_high, cell_coords[..., 1])
  phys_est_z, c = trilerp(comp, vtx_low, vtx_high, cell_coords[..., 2])
  return np.array([phys_est_x, phys_est_y, phys_est_z]), np.array([a,b,c])

# F
def diff_phys(comp: np.ndarray, phys: np.ndarray, vtx_low, vtx_high, cell_coords):
  phys_est, coeff = comp2phys(comp, vtx_low, vtx_high, cell_coords)
  return phys - phys_est

def diff_phys_dim(comp: np.ndarray, phys: np.ndarray, vtx_low, vtx_high, cell_coords_dim):
  phys_est = trilerp(comp, vtx_low, vtx_high, cell_coords_dim)
  return phys - phys_est
# op.root_scalar(diff_phys, method='newton')

def diff_phys_dim_inv_jac():
  pass

class CurviInterpField:
  def __init__(self):
    pass

class CurviInterpCell:
  def __init__(self, min_comp, max_comp, phys_coords):
    self.min_comp = np.array(min_comp)
    self.max_comp = np.array(max_comp)
    self.phys = phys_coords
    self.init_comp = min_comp + (max_comp - min_comp) / 2
    self.coeff = None

    # xs = [min_comp[0], max_comp[0]]
    # ys = [min_comp[1], max_comp[1]]
    # zs = [min_comp[2], max_comp[2]]
    # self.interp = RegularGridInterpolator((xs, ys, zs), phys_coords)

  # Newton's method to find computational coordinate given physical cooridnate
  def phys2comp(self, phys: np.ndarray, init_comp=None, tol=1.48e-8, maxiter=50, rtol=0.0):
    if init_comp is None:
      init_comp = self.init_comp

    comp = init_comp
    goal_diff = np.zeros(3)
    print(f"comp low high: {self.min_comp} {self.max_comp}")
    for i in range(maxiter):
      phys_est, coeff = self.comp2phys(comp)
      diff_funcval = (phys_est - phys)

      jac_inv = self.get_jacobian_inv(comp, coeff)

      diff_comp = (jac_inv @ diff_funcval.reshape(3, 1)).flatten()
      new_comp = comp - diff_comp
      # print(f"    estimate: {new_comp}\n    error: {np.abs(diff_funcval-goal_diff)}")
      # print("My Jac inv\n", jac_inv, diff_funcval, phys_est)
      # print(f" diff_comp, comp: {diff_comp} {comp}\n")
      # print(phys_est.shape, coeff.shape, phys.shape)
      # print(new_comp.shape, comp.shape)
      # print(jac_inv.shape, diff_funcval.shape)
      # print(f'coeff: {coeff[:, 0]}')
      print(f'phys_est: {phys_est} comp: {new_comp}  error: {np.abs(diff_funcval-goal_diff)}')
      # if np.all(np.isclose(new_comp, comp, rtol=rtol, atol=tol)):
      if np.all(np.isclose(diff_funcval, goal_diff, rtol=rtol, atol=tol)):
        print(f'iter {i}')
        return new_comp
      comp = new_comp
    return comp

  # def phys2comp(self, phys):
  #   return self.find_comp(phys)

  def comp2phys(self, comp):
    phys_est, coeff = trilerp(comp, self.min_comp, self.max_comp, self.phys)
    # phys_est_y, b = trilerp(comp, self.min_comp, self.max_comp, self.phys[..., 1])
    # phys_est_z, c = trilerp(comp, self.min_comp, self.max_comp, self.phys[..., 2])
    # self.coeff = np.array([a,b,c])
    # return np.array([phys_est_x, phys_est_y, phys_est_z]), np.array([a,b,c])
    # print(phys_est.shape, coeff.shape)
    # self.interp(comp)
    return phys_est, coeff
  
  def get_jacobian(self, comp, coeff):
    a,b,c = coeff
    # print("COE: \n", coeff.T)
    # print("My COMP: \n", comp)
    return np.array([
      [
        a[1] + a[4]*comp[1] + a[5]*comp[2] + a[7]*comp[1]*comp[2],
        a[2] + a[4]*comp[0] + a[6]*comp[2] + a[7]*comp[0]*comp[2],
        a[3] + a[5]*comp[0] + a[6]*comp[1] + a[7]*comp[0]*comp[1],
      ],
      [
        b[1] + b[4]*comp[1] + b[5]*comp[2] + b[7]*comp[1]*comp[2],
        b[2] + b[4]*comp[0] + b[6]*comp[2] + b[7]*comp[0]*comp[2],
        b[3] + b[5]*comp[0] + b[6]*comp[1] + b[7]*comp[0]*comp[1],
      ],
      [
        c[1] + c[4]*comp[1] + c[5]*comp[2] + c[7]*comp[1]*comp[2],
        c[2] + c[4]*comp[0] + c[6]*comp[2] + c[7]*comp[0]*comp[2],
        c[3] + c[5]*comp[0] + c[6]*comp[1] + c[7]*comp[0]*comp[1],
      ]
    ])
  
  def get_jacobian_inv(self, comp, coeff):
    # print(comp)
    comp = (comp - self.min_comp) / (self.max_comp - self.min_comp)
    # print(comp)
    
    jac = self.get_jacobian(comp, coeff)
    np_det = np.linalg.det(jac)
    # np_inv = np.linalg.inv(jac)
    # det = (-jac[0,0]*jac[1,1]*jac[2,2] - jac[0,1]*jac[1,2]*jac[2,0] - jac[0,2]*jac[1,0]*jac[2,1] + 
    #         jac[0,2]*jac[1,1]*jac[2,0] + jac[0,1]*jac[1,0]*jac[2,2] + jac[0,0]*jac[1,2]*jac[2,1])
    det = np_det
    inv = np.array([
      [
        jac[1,1]*jac[2,2] - jac[1,2]*jac[2,1],
        -jac[0,1]*jac[2,2] + jac[0,2]*jac[2,1],
        jac[0,1]*jac[1,2] - jac[0,2]*jac[1,1],
      ],
      [
        -jac[1,0]*jac[2,2] + jac[1,2]*jac[2,0],
        jac[0,0]*jac[2,2] - jac[0,2]*jac[2,0],
        -jac[0,0]*jac[1,2] + jac[0,2]*jac[1,0],
      ],
      [
        jac[1,0]*jac[2,1] - jac[1,1]*jac[2,0],
        jac[0,0]*jac[2,1] - jac[0,1]*jac[2,0],
        jac[0,0]*jac[1,1] - jac[0,1]*jac[1,0],
      ],
    ])
    
    # print("jac, np_inv, inv\n", jac)
    # print(np_inv, inv/det)
    # print(inv)
    # print(inv/det)
    # print(np_det)
    # print(f"det {-det}")
    det = np_det
    inv = inv/det
    # print("My Jac inv\n", jac, "\n", inv, det, "\n")
    return inv