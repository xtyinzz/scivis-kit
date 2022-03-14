// #include <grid.h>
#include "container/field.h"
#include "container/solution.h"
#include "common.h"

#include <netcdf>

#include <string>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

class Func1 {
  private:
    float a, b;
  public:
    Func1() {}
    Func1(float a, float b): a(a), b(b) {}
    float eval(float x, float y, float z) {
      float t1 = x*x + ((1+b)*y)*((1+b)*y) + z*z - 1;
      float t2 = -x*x*z*z*z -a*y*y*z*z*z;
      return t1*t1*t1 + t2;
    }
};

struct Coord {
  float x, y, z;
};

int main() {


  // setup Field
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
  Solution<float> sol(g.getDimLen(0), g.getDimLen(1), g.getDimLen(2));
  sol.initData();

  Field<float> f(&g, &sol);
  std::cout << f.getDimLen(0) << "---" << f.g->getDimLen(0) << std::endl;

  // fill in data
  std::ifstream fpos("data/test/task1_plane.txt", std::ios::in);
  
  std::string s;
  float a, b;
  std::getline(fpos, s);
  std::vector<std::string> line = strSplit(s, ',');
  a = std::stof(line[0]);
  b = std::stof(line[1]);

  Func1 fn(a, b);
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
        // std::cout << i << " " << j << " " << k << " isEqual? " << (int) (val == f.getVal(x, y, z)) << '\n';
      }
    }
  }
  std::cout << "s length: " << f.s->length << " f length: " << f.getDimLen(0) * f.getDimLen(1) * f.getDimLen(2) << '\n';
  
  std::getline(fpos, s);
  line = strSplit(s, ',');
  int xlen = std::stoi(line[0]);
  int ylen = std::stoi(line[1]);

  // query the values
  std::vector<float> vals(xlen*ylen, 0);
  int i = 0;
  while (std::getline(fpos, s)) {
    line = strSplit(s, ',');
    float x = std::stof(line[0]);
    float y = std::stof(line[1]);
    float z = std::stof(line[2]);
    float val = f.getVal(x, y, z);
    vals[i] = val;
    i++;
  }
  fpos.close();

  // netCDF I/O
  std::string ncFileName = "data/sub/task1_plane_value.nc";

  netCDF::NcFile dataFile(ncFileName.c_str(), netCDF::NcFile::replace);

  auto xDim = dataFile.addDim("x", xlen);
  auto yDim = dataFile.addDim("y", ylen);
  auto data = dataFile.addVar("val", netCDF::ncFloat, {xDim, yDim,});

  data.putVar(vals.data());
  printf("*** SUCCESS writing file %s!\n", ncFileName.c_str());

}