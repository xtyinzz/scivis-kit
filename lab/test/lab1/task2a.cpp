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
  int dimLen = 256;
  Solution<float> solution(dimLen, dimLen, dimLen);
  solution.load(pdata);
  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 256; j++) {
      for (int k = 0; k < 256; k++) {
        float myval = solution.val(i, j, k);
        float dval = solution.data[i*256*256+j*256+k];
        if (myval != dval) std::cout << myval << " VS " << dval <<" at " << i <<"-"<<j<<"-"<<k<<"\n";
      }
    }
  }
  // data.save(pout);

  std::string ncFileName = "data/sub/task2a.nc";

  netCDF::NcFile dataFile(ncFileName.c_str(), netCDF::NcFile::replace);

  auto xDim = dataFile.addDim("x", dimLen);
  auto yDim = dataFile.addDim("y", dimLen);
  auto zDim = dataFile.addDim("z", dimLen);
  auto data = dataFile.addVar("val", netCDF::ncFloat, {xDim, yDim, zDim});

  data.putVar(solution.data.data());
  printf("*** SUCCESS writing file %s!\n", ncFileName.c_str());

}