// #include <grid.h>
#include "container/field.h"
#include "container/solution.h"
#include <netcdf.h>
#include <netcdf>

#include <string>
#include <cmath>
#include <string>
#include <iostream>
#include <vector>

class Func1 {
  private:
    float a, b;
  public:
    Func1() {}
    Func1(float a, float b): a(a), b(b) {}
    float eval(float x, float y, float z) {
      float t1 = x*x + ((1+b)*y)*((1+b)*y) + z*z - 1;
      float t2 = -x*x*z*z*z - a*y*y*z*z*z;
      return t1*t1*t1 + t2;
    }
};

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

int main()
{
  // setup Grid
  int dimLen = 300;
  float xmin = -1;
  float xmax = 1;
  float ymin = -1;
  float ymax = 1;
  float zmin = -1;
  float zmax = 1;
  float dx = (xmax - xmin) / (dimLen - 1);
  float dy = (xmax - xmin) / (dimLen - 1);
  float dz = (xmax - xmin) / (dimLen - 1);

  Grid g(xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz);
  Solution<float> s(g.getDimLen(0), g.getDimLen(1), g.getDimLen(2));
  s.initData();

  Field<float> f(&g, &s);
  std::cout << f.getDimLen(0) << "---" << f.g->getDimLen(0) << std::endl;

  Func1 fn(1, 1);
  for (int i = 0; i < f.getDimLen(0); i++)
  {
    for (int j = 0; j < f.getDimLen(1); j++)
    {
      for (int k = 0; k < f.getDimLen(2); k++)
      {
        float x = xmin + i * dx;
        float y = ymin + j * dy;
        float z = zmin + k * dz;
        float val = fn.eval(x, y, z);
        f.setVal(i, j, k, val);
        // std::cout << i << " " << j << " " << k << " isEqual? " << (int) (val == f.val(x, y, z)) << '\n';
      }
    }
  }
  std::cout << "s length: " << f.s->length << " f length: " << f.getDimLen(0) * f.getDimLen(1) * f.getDimLen(2) << '\n';

  // netCDF I/O (C version)
  /* This will be the netCDF ID for the file and data variable. */
  int ncid, varid;

  std::string ncFileName = "data/sub/task1a.nc";
  /* Loop indexes, and error handling. */
  int x, y, z, retval, x_dimid, y_dimid, z_dimid;

  if ((retval = nc_create(ncFileName.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid)))
    ERR(retval);

  /* Define the dimensions in the root group. Dimensions are visible
   * in all subgroups. */
  if ((retval = nc_def_dim(ncid, "x", dimLen, &x_dimid)))
    ERR(retval);

  if ((retval = nc_def_dim(ncid, "y", dimLen, &y_dimid)))
    ERR(retval);

  if ((retval = nc_def_dim(ncid, "z", dimLen, &z_dimid)))
    ERR(retval);

  int dimids[3] = { x_dimid, y_dimid, z_dimid };
  if ((retval = nc_def_var(ncid, "val", NC_FLOAT, 3,
                          dimids, &varid)))
    ERR(retval);

  if ((retval = nc_put_var_float(ncid, varid, f.s->data.data())))
    ERR(retval);
         
  /* Close the file, freeing all resources. */
  if ((retval = nc_close(ncid)))
    ERR(retval);

  printf("*** SUCCESS writing file %s!\n", ncFileName.c_str());
  return 0;
}