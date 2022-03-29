#ifndef SVK_COMMON_IO
#define SVK_COMMON_IO

#include "util.h"
#include "container/solution.h"
#include <glm/glm.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

using namespace Eigen;

// read .vec file with 3 integers of dimensions and floats for the rest:
// as std::vector<Eigen::Vector3f>
// .vec file:
//   xdim ydim zdim
//   x1 y1 z1
//   x2 y2 z2
//   ...
Solution<Vector3f> readVec(const std::string &fpath, bool verbose=true) {
  Solution<Vector3f> data;
  data.fromVec(fpath, verbose);
  return data;
}

void writeVec(const std::string &fpath, int xlen, int ylen, int zlen, const std::vector<Vector3f> &vecs) {
  std::ofstream fdata(fpath, std::ios::binary);
  fdata.write(reinterpret_cast<const char*>(&xlen), sizeof(int));
  fdata.write(reinterpret_cast<const char*>(&ylen), sizeof(int));
  fdata.write(reinterpret_cast<const char*>(&zlen), sizeof(int));
  fdata.write(reinterpret_cast<const char*>(vecs.data()), sizeof(Vector3f) * vecs.size());
  printf("writeVec complete: %i (%ix%ix%i) 3D float vector is written to %s\n",
        (int)vecs.size(), xlen, ylen, zlen, fpath.c_str());
  fdata.close();
}

std::vector<std::vector<float>> readGrid(const std::string &fpath) {
  std::ifstream fdata(fpath, std::ios::binary);
  if (fdata.fail()) {
    printf("File reading failed (%s)", fpath.c_str());
  }

  Vector3<int> dims;
  fdata.read(reinterpret_cast<char*>(dims.data()), sizeof(int)*3);
  int ndata = dims.prod();

  std::vector<float> xCoord(dims(0));
  std::vector<float> yCoord(dims(1));
  std::vector<float> zCoord(dims(2));
  fdata.read(reinterpret_cast<char*>(xCoord.data()), dims(0)*sizeof(float));
  fdata.read(reinterpret_cast<char*>(yCoord.data()), dims(1)*sizeof(float));
  fdata.read(reinterpret_cast<char*>(zCoord.data()), dims(2)*sizeof(float));

  printf("readGrid complete: 3 array %4i, %4i, %4i floats are read from %s\n",
          dims(0), dims(1), dims(2), fpath.c_str());
  fdata.close();
  
  return std::vector<std::vector<float>>{xCoord, yCoord, zCoord};
}

void writeGrid(const std::string &fpath, const VectorXf &x, const VectorXf &y, const VectorXf &z) {

}

#endif