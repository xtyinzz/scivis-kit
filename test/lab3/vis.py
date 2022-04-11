import numpy as np
from matplotlib import pyplot as plt

# produce rk1 image
with open("data/sl_len_rk1.raw", "rb") as f:
  sllens_bytes = f.read()
with open("data/sl_rk1.raw", "rb") as f:
  sl_coords_bytes = f.read();
  
sllens = np.frombuffer(sllens_bytes, dtype=np.int32)
sl_coords = np.frombuffer(sl_coords_bytes, dtype=np.float32)

sls = []
for sllen in sllens:
  sls.append(sl_coords[:sllen*3].reshape(sllen, 3))
  sl_coords = sl_coords[sllen*3:]

fig = plt.figure()
ax = plt.axes(projection='3d')
for sl in sls:
  ax.plot(sl[:, 0], sl[:, 1], sl[:, 2])
plt.savefig("img/traces_rk1.png")
print("Images writte to img/traces_rk1")


# produce rk4 image
with open("data/sl_len_rk4.raw", "rb") as f:
  sllens_bytes = f.read()
with open("data/sl_rk4.raw", "rb") as f:
  sl_coords_bytes = f.read();
  
sllens = np.frombuffer(sllens_bytes, dtype=np.int32)
sl_coords = np.frombuffer(sl_coords_bytes, dtype=np.float32)

sls = []
for sllen in sllens:
  sls.append(sl_coords[:sllen*3].reshape(sllen, 3))
  sl_coords = sl_coords[sllen*3:]

fig = plt.figure()
ax = plt.axes(projection='3d')
for sl in sls:
  ax.plot(sl[:, 0], sl[:, 1], sl[:, 2])
plt.savefig("img/traces_rk4.png")
print("Images writte to img/traces_rk4")