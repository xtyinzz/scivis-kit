// #include <grid.h>
#include "container/field.h"

#include <netcdf>

#include <string>
#include <cmath>
#include <string>
#include <iostream>

class Func1 {
  private:
    float a, b;
  public:
    Func1() {}
    Func1(float a, float b): a(a), b(b) {}
    float eval(float x, float y, float z) {
      float t1 = x*x + ((1+b)*y)*((1+b)*y) + z*z - 1;
      float t2 = x*x*z*z*z - a*y*y*z*z*z;
      return t1*t1*t1 - t2;
    }
};

int main() {
  std::string pdata = "data/raw/resampled_256^3.raw";
  Solution<float> solution(256, 256, 256);
  solution.load(pdata);
  Grid g(256, 256, 256);
  Field<float> field(&g, &solution);
  

  std::ifstream fpos("data/test/task2_plane.txt", std::ios::in);
  
  std::string s;
  std::getline(fpos, s);
  std::vector<std::string> line = strSplit(s, ',');

  int xlen, ylen;
  xlen = std::stoi(line[0]);
  ylen = std::stoi(line[1]);

  // query the values
  std::vector<float> vals(xlen*ylen, 0);
  int i = 0;
  while (std::getline(fpos, s)) {
    line = strSplit(s, ',');
    float x = std::stof(line[0]);
    float y = std::stof(line[1]);
    float z = std::stof(line[2]);
    float val = field.getVal(z, y, x);
    vals[i] = val;
    i++;
  }
  fpos.close();

  std::cout << i << " VS " << xlen*ylen << "\n";

  // netCDF I/O
  std::string ncFileName = "data/sub/task2_plane_value.nc";

  netCDF::NcFile dataFile(ncFileName.c_str(), netCDF::NcFile::replace);

  auto xDim = dataFile.addDim("x", xlen);
  auto yDim = dataFile.addDim("y", ylen);
  auto data = dataFile.addVar("val", netCDF::ncFloat, {xDim, yDim,});

  data.putVar(vals.data());
  printf("*** SUCCESS writing file %s!\n", ncFileName.c_str());

}