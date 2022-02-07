// #include <grid.h>
#include "field.h"
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

struct Coord {
  float x, y, z;
};

std::vector<std::string> strSplit(std::string s, char delim) {
  std::vector<std::string> results;
  std::stringstream ss(s);
  std::string str;
  while (getline(ss, str, delim)) {
    results.push_back(str);
  }
  return results;
}

int main() {
  std::ifstream fpos("test/task1_random.txt", std::ios::in);
  
  std::string s;
  std::getline(fpos, s);
  std::vector<std::string> line = strSplit(s, ',');

  float a, b;
  a = std::stof(line[0]);
  b = std::stof(line[1]);


  std::vector<Coord> pos;
  while (std::getline(fpos, s)) {
    line = strSplit(s, ',');
    Coord coord = { std::stof(line[0]), std::stof(line[1]), std::stof(line[2]) };
    pos.push_back(coord);
  }
  fpos.close();

  std::ofstream fval("sub/task1_random_value.txt", std::ios::out);
  Func1 fn(a, b);
  for (int i = 0; i < pos.size(); i++) {
    Coord coord = pos[i];
    float val = fn.eval(coord.x, coord.y, coord.z);
    fval << val << std::endl;
  }
  fval.close();

}