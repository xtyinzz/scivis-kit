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
  std::string pdata = "data/raw/resampled_256^3.raw";
  std::string pout = "data/raw/task2a.raw";
  Solution<float> data(256, 256, 256);
  data.load(pdata);
  Grid g(256, 256, 256);
  Field<float> field(&g, &data);
  

  std::ifstream fpos("data/test/task2_random.txt", std::ios::in);
  
  std::string s;
  std::vector<std::string> line;

  // int xlen, ylen, zlen;
  // std::getline(fpos, s);
  // line = strSplit(s, ',');
  // xlen = std::stoi(line[0]);
  // ylen = std::stoi(line[1]);


  std::ofstream fval("data/sub/task2_random_value.txt", std::ios::out);
  while (std::getline(fpos, s)) {
    line = strSplit(s, ',');
    float x = std::stof(line[0]);
    float y = std::stof(line[1]);
    float z = std::stof(line[2]);
    float val = field.val(x, y, z);
    std::cout << x <<' '<<y<<' '<<z<<' ' <<val << "\n";
    fval << val << std::endl;
  }
  fpos.close();
  fval.close();
}