#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <string>
#include <sstream>

// product between vector elements
template <typename T>
T product(std::vector<T> vec) {
  T prod = 1;
  for(std::size_t i = 0; i < vec.size(); i++) {
    prod *= vec[i];
  }
  return prod;
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
          T alpha, T beta, T gamma) {
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

std::vector<std::string> strSplit(std::string s, char delim) {
  std::vector<std::string> results;
  std::stringstream ss(s);
  std::string str;
  while (getline(ss, str, delim)) {
    results.push_back(str);
  }
  return results;
}

#endif