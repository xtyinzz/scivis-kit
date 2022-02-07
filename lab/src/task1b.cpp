// #include <grid.h>
#include "field.h"
#include "util.h"
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
      float t2 = x*x*z*z*z - a*y*y*z*z*z;
      return t1*t1*t1 - t2;
    }
};

int main() {
  std::ifstream fpos("data/test/task1_random.txt", std::ios::in);
  
  std::string s;
  std::getline(fpos, s);
  std::vector<std::string> line = strSplit(s, ',');

  float a, b;
  a = std::stof(line[0]);
  b = std::stof(line[1]);

  std::ofstream fval("data/sub/task1_random_value.txt", std::ios::out);
  Func1 fn(a, b);
  while (std::getline(fpos, s)) {
    line = strSplit(s, ',');
    float x = std::stof(line[0]);
    float y = std::stof(line[1]);
    float z = std::stof(line[2]);
    float val = fn.eval(x, y, z);
    std::cout << x << ' ' << y << ' ' << z << ' ' << val << '\n';
    fval << val << std::endl;
  }
  fpos.close();
  fval.close();

}