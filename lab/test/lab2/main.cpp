#include "renderer/VolumeRenderer.h"
#include <string>
#include <iostream>
#include <glm/vec4.hpp>
#include "Eigen/Dense"
 
// using Eigen::MatrixXd;

int main() {
  VolumeRenderer vr;
  glm::vec4 vec(1,2,3,4);

  std::cout << vr.tmp << "\n" << vr.tf.tmp << std::endl;

  // MatrixXd m(2,2);
  // m(0,0) = 3;
  // m(1,0) = 2.5;
  // m(0,1) = -1;
  // m(1,1) = m(1,0) + m(0,1);
  // std::cout << m << std::endl;

}