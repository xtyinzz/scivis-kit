#ifndef SVK_COMMON_UTIL
#define SVK_COMMON_UTIL

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <sstream>

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

#endif