// #include <grid.h>
#include "field.h"
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
  // std::string pdata = "data/resampled_256^3.raw";
  // std::string pout = "data/test.raw";
  // int length = 256 * 256 * 256;
  // Solution<float> data(pout, length, 32);
  // data.save(pout);

  // setup Grid
  float xmin=-1;
  float xmax=1;
  float ymin=-1;
  float ymax=1;
  float zmin=-1;
  float zmax=1;
  float dx = 0.02;
  float dy = 0.02;
  float dz = 0.02;
  std::string fp = "data/func.raw";

  Grid g(xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz);
  Solution<float> s(g.dimLength(0), g.dimLength(1), g.dimLength(2));
  // s.load(fp);

  Field<float> f(&g, &s);
  std::cout << f.dimLength(0) <<  "---" << f.g->dimLength(0) << std::endl;

  Func1 fn(1, 1);
  for (int i = 0; i < f.dimLength(0); i++) {
    for (int j = 0; j < f.dimLength(1); j++) {
      for (int k = 0; k < f.dimLength(2); k++) {
        float x = xmin + i*dx;
        float y = ymin + j*dy;
        float z = zmin + k*dz;
        float val = fn.eval(x, y, z);
        f.setVal(i, j, k, val);
        // std::cout << i << " " << j << " " << k << " isEqual? " << (int) (val == f.val(x, y, z)) << '\n';
      }
    }
  }

  f.s->save(fp);

  std::cout << "s length: " << f.s->length << " f length: " << f.dimLength(0)*f.dimLength(1)*f.dimLength(2) << '\n';
}