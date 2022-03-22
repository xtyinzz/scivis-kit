#include "container/field.h"
#include "container/grid_rectlinear.h"
// #include "common.h"
#include <glm/glm.hpp>

#include <string>
#include <vector>
#include <iostream>

#include <Eigen/Dense>

using namespace Eigen;

// using Eigen::MatrixXd;
int main() {
  std::vector<double> coord = {1., 2., 3.};
  RectlinearGrid rg(&coord, &coord, &coord);
  // std::cout << rg.getDomain(0)[0] << "\n";
  // std::cout << rg.getDomain(1)[1] << "\n";
  // std::cout << rg.getDomain(2)[1] << "\n";

  Field<float> f(&rg);
  std::cout << f.getDimExtent(0)[0] << "\n";
  std::cout << f.getDimExtent(0)[1] << "\n";



  MatrixXd M = MatrixXd::Random(3,8);
  MatrixXd r = M.row(0);
  std::cout << M << "\n";
  std::cout << r << "\n";
  std::cout << M(0) << "\n";
  std::cout << M(0,1) << "\n";
  std::cout << r(0) << "\n";

}