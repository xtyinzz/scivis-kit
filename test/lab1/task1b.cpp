// #include <grid.h>
#include "container/field.h"
#include "common.h"
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
      float t2 = -x*x*z*z*z - a*y*y*z*z*z;
      return t1*t1*t1 + t2;
    }
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
  std::ifstream fpos("data/test/task1_random.txt", std::ios::in);
  
  std::string s;
  std::getline(fpos, s);
  std::vector<std::string> line = strSplit(s, ',');

  float a, b;
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
        // std::cout << i << " " << j << " " << k << " isEqual? " << (int) (val == f.val(x, y, z)) << '\n';
      }
    }
  }
  std::cout << "s length: " << f.s->length << " f length: " << f.getDimLen(0) * f.getDimLen(1) * f.getDimLen(2) << '\n';

  // evaluate and write data
  std::ofstream fval("data/sub/task1_random_value.txt", std::ios::out);
  while (std::getline(fpos, s)) {
    line = strSplit(s, ',');
    float x = std::stof(line[0]);
    float y = std::stof(line[1]);
    float z = std::stof(line[2]);
    float val = f.getVal(x, y, z);
    std::cout << x << ' ' << y << ' ' << z << ' ' << val << '\n';
    fval << val << std::endl;
  }
  fpos.close();
  fval.close();

}