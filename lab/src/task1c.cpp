// #include <grid.h>
#include "field.h"
#include "solution.h"
#include <netcdf.h>
#include <string>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace netCDF;

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

// timesteps of data.
#define NDIMS    3
#define NX 500
#define NY 500
#define NZ 1

// Names of things. 
#define LVL_NAME "X"
#define LAT_NAME "Y"
#define LON_NAME "Z"
#define REC_NAME "time"
#define PRES_NAME     "pressure"
#define TEMP_NAME     "temperature"
#define MAX_ATT_LEN  80
// These are used to construct some example data. 
#define SAMPLE_PRESSURE 900
#define SAMPLE_TEMP     9.0
#define START_LAT       25.0
#define START_LON       -125.0


int main() {
  std::ifstream fpos("test/task1_plane.txt", std::ios::in);
  
  std::string s;
  std::getline(fpos, s);
  std::vector<std::string> line = strSplit(s, ',');

  float a, b;
  a = std::stof(line[0]);
  b = std::stof(line[1]);

  int xlen, ylen, zlen;
  std::getline(fpos, s);
  line = strSplit(s, ',');
  xlen = std::stoi(line[0]);
  ylen = std::stoi(line[1]);
  zlen = 1;

  NcFile test(FILE_NAME, NcFile::replace);




  std::ofstream fval("test/task1_plane_value.txt", std::ios::out);
  Solution<float> solution(xlen, ylen, zlen);
  Func1 fn(a, b);
  while (std::getline(fpos, s)) {
    line = strSplit(s, ',');
    float x = std::stof(line[0]);
    float y = std::stof(line[1]);
    float z = std::stof(line[2]);
    float val = fn.eval(x, y, z);
    fval << val << std::endl;
  }
  fpos.close();
  fval.close();

}