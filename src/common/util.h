#ifndef SVK_COMMON_UTIL
#define SVK_COMMON_UTIL

#include <vector>
#include <string>
#include <sstream>

#include <glm/glm.hpp>
#include <Eigen/Dense>

using namespace Eigen;

template <typename T>
T product(std::vector<T> vec) {
  T prod = 1;
  for(std::size_t i = 0; i < vec.size(); i++) {
    prod *= vec[i];
  }
  return prod;
}

template <typename T>
T sum(std::vector<T> vec) {
  T res = 0;
  for(std::size_t i = 0; i < vec.size(); i++) {
    res += vec[i];
  }
  return res;
}

std::vector<std::string> strSplit(std::string s, char delim) {
  std::vector<std::string> results;
  std::stringstream ss(s);
  std::string str;
  while (getline(ss, str, delim)) {
    results.push_back(str);
  }
  return results;
}

void printVec( glm::vec3 vec )
{
	printf("  %7.4f %7.4f %7.4f\n", vec[0], vec[1], vec[2] );
}

void printVec( glm::vec4 vec )
{
	printf("  %7.4f %7.4f %7.4f %7.4f\n", vec[0], vec[1], vec[2], vec[3] );
}


template <typename T=Vector3f>
void swapVecDim(std::vector<Vector3f> &vecs, int idim, int jdim, int kdim) {
  for (int i = 0; i < vecs.size(); i++) {
    T thisvec = vecs[i];
    T tmpvec = thisvec;
    thisvec(0) = tmpvec(idim);
    thisvec(1) = tmpvec(jdim);
    thisvec(2) = tmpvec(kdim);
    vecs[i] = thisvec;
  }
}

#endif