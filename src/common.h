#ifndef UTIL_H
#define UTIL_H

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <sstream>

#define EPS 1e-8

// product between vector elements
template <typename T>
T product(std::vector<T> vec) {
  T prod = 1;
  for(std::size_t i = 0; i < vec.size(); i++) {
    prod *= vec[i];
  }
  return prod;
}


//		  ________
//		0        1
template <typename T>
T lerp(T v0, T v1, T alpha) {
  T p = (1-alpha)*v0 + v1*alpha;
  return p;
}

//		  ________
//		0        1


template <typename T>
T lerpGLM(float x, float x0, float x1, T v0, T v1) {
  float alpha = (x-x0) / (x1 - x0);
  T p = (1-alpha)*v0 + v1*alpha;
  return p;
}

template <typename T>
T lerp(float x, float x0, float x1, T v0, T v1) {
  float alpha = (x-x0) / (x1 - x0);
  T p = (1-alpha)*v0 + v1*alpha;
  return p;
}


//		  2  ________ 3
//		   /        /
//		  /        /
//		 /________/
//		0        1
template <typename T>
T bilerp(T v0, T v1, T v2, T v3, T alpha, T beta) {
  T p = (1-alpha)*(1-beta)*v0 + 
        alpha*(1-beta)*v1 + 
        (1-alpha)*beta*v2 + 
        alpha*beta*v3;
  return p;
}

  //
  //		    6________7
  //		   /|       /|
  //		  / |      / |
  //		4/_______5/  |
  //		|  2|___ |___|3
  //		|  /     |  /
  //		| /      | /
  //		|/_______|/
  //		0        1
template <typename T>
T trilerp(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7,
          float alpha, float beta, float gamma) {
  T p = (1-alpha)*(1-beta)*(1-gamma)*v0 + 
        alpha*(1-beta)*(1-gamma)*v1 + 
        (1-alpha)*beta*(1-gamma)*v2 + 
        alpha*beta*(1-gamma)*v3 + 
        (1-alpha)*(1-beta)*gamma*v4 + 
        alpha*(1-beta)*gamma*v5 + 
        (1-alpha)*beta*gamma*v6 +
        alpha*beta*gamma*v7;
  return p;
}
// template <typename T>


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