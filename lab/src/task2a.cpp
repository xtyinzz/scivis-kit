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
  data.save(pout);
}