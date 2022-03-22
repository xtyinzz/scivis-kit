import torch
import torch.nn as nn
from util.utils import report_gpumem, cpuStats

class PositionalEncoding(nn.Module):
  def __init__(self, emb_dim=64):
    super(PositionalEncoding, self).__init__()
    self.coeff = torch.pi * 2 ** torch.arange(emb_dim//2)
  
  def forward(self, x):
    # x: (b, n, c)
    coeff = self.coeff.to(x.device)
    # add dimensions for mulitplication broadcast
    for dim_i in range(len(x.shape)):
      coeff = coeff.unsqueeze(0) # (1, 1, 1, emb_dim//2)

    x_expand = x.unsqueeze(-1) # (b, n, c, 1)
    s = torch.sin(x_expand * coeff)
    c = torch.cos(x_expand * coeff)
    x_pe = torch.concat([s,c], dim=-1) # (b, n, c, emb_dim)
    x_pe = x_pe.view(*x.shape[:-1], -1) # (b, n, c*emb_dim)
    return x_pe

class LinearLayer(nn.Module):
  def __init__(self, in_dim, out_dim=256, act_fn=nn.ReLU(), use_norm=False) -> None:
    super(LinearLayer, self).__init__()
    self.layer = nn.ModuleList([ nn.Linear(in_dim, out_dim) ])
    if use_norm:
      self.layer.append(nn.LayerNorm(out_dim))
    
    if act_fn is not None:
      self.layer.append(act_fn)
    
    self.layer = nn.Sequential(*self.layer)
    
  def forward(self, x):
    return self.layer(x)

# Fourrier Feature Network: PE+MLP
class MLP(nn.Module):
  '''
  if pe_dim = 0:
    pe_dim = 1st hidden_dim
  '''
  def __init__(self, in_dim, hidden_dims, out_dim, pe_dim=0):
    super(MLP, self).__init__()
    self.net = nn.ModuleList([])

    if pe_dim is not None:
      # PE dimension = hidden_dims[0] if not specified
      pe_dim = hidden_dims[0] if pe_dim == 0 else pe_dim*in_dim
      self.net.append(PositionalEncoding(pe_dim // in_dim))
      # update input dimension for following linear layers
      in_dim = pe_dim

    # intermediate layers
    self.net.append( LinearLayer(in_dim, hidden_dims[0], use_norm=True) )
    for i in range(1, len(hidden_dims)):
      self.net.append(
        LinearLayer(hidden_dims[i-1], hidden_dims[i], use_norm=True)
      )
    # final layer with Tanh
    self.net.append(
      LinearLayer(hidden_dims[-1], out_dim, act_fn=nn.Tanh())
    )
    
    self.net = nn.Sequential(*self.net)

  def forward(self, x):
    # print(x.shape)
    # print("************* GPU *************")
    # report_gpumem()
    # print("************* CPU *************")
    # cpuStats()
    return self.net(x)

# Fourrier Feature Network: PE+MLP
class ResMLP(nn.Module):
  def __init__(self, in_dim, hidden_dims, out_dim, pe_dim=64):
    super(ResMLP, self).__init__()
    self.use_pe = pe_dim is not None
    self.net = nn.ModuleList([])

    self.pe = PositionalEncoding(hidden_dims[0] // in_dim)
    if pe_dim is not None:
      in_dim = hidden_dims[0]

    # intermediate layers
    hidden_dims.insert(0, in_dim)
    for i in range(len(hidden_dims)-1):
      self.net.append(
        LinearLayer(hidden_dims[i], hidden_dims[i+1], use_norm=True)
      )
    # final layer with Tanh
    self.final = LinearLayer(hidden_dims[-1], out_dim, act_fn=None)

  def forward(self, x):
    if self.use_pe:
      x = self.pe(x)
    
    for layer in self.net:
      x = x + layer(x)

    return self.final(x)


# Author: Vincent Sitzmann
# Source: https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb
#
class SineLayer(nn.Module):
  # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
  
  # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
  # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
  # hyperparameter.
  
  # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
  # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
  
  def __init__(self, in_features, out_features, bias=True,
              is_first=False, omega_0=30):
    super().__init__()
    self.omega_0 = omega_0
    self.is_first = is_first
    
    self.in_features = in_features
    self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    self.init_weights()
  
  def init_weights(self):
    with torch.no_grad():
      if self.is_first:
        self.linear.weight.uniform_(-1 / self.in_features,
                                      1 / self.in_features)
      else:
        self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                        np.sqrt(6 / self.in_features) / self.omega_0)
      
  def forward(self, input):
    return torch.sin(self.omega_0 * self.linear(input))
    

# Author: Vincent Sitzmann
# Source: https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb
#
class Siren(nn.Module):
  def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, 
              first_omega_0=30, hidden_omega_0=30.):
    super().__init__()

    self.net = []
    self.net.append(SineLayer(in_features, hidden_features, 
                    is_first=True, omega_0=first_omega_0))

    for i in range(hidden_layers):
      self.net.append(SineLayer(hidden_features, hidden_features, 
                      is_first=False, omega_0=hidden_omega_0))

    if outermost_linear:
      final_linear = nn.Linear(hidden_features, out_features)
        
      with torch.no_grad():
          final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                        np.sqrt(6 / hidden_features) / hidden_omega_0)
            
      self.net.append(final_linear)
    else:
      self.net.append(SineLayer(hidden_features, out_features, 
                                is_first=False, omega_0=hidden_omega_0))
    
    self.net = nn.Sequential(*self.net)

  def forward(self, coords):
    coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
    output = self.net(coords)
    return output, coords