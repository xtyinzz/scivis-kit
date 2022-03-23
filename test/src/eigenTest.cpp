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
  // std::vector<double> coord = {1., 2., 3.};
  // RectlinearGrid rg(&coord, &coord, &coord);
  // // std::cout << rg.getDomain(0)[0] << "\n";
  // // std::cout << rg.getDomain(1)[1] << "\n";
  // // std::cout << rg.getDomain(2)[1] << "\n";

  // Field<float> f(&rg);
  // std::cout << f.getDimExtent(0)[0] << "\n";
  // std::cout << f.getDimExtent(0)[1] << "\n";



  MatrixXd M = MatrixXd::Random(3,8);
  MatrixXd r = M.row(0);
  std::cout << M << "\n";
  std::cout << r << "\n";
  std::cout << M(0) << "\n";
  std::cout << M(0,1) << "\n";
  std::cout << r(0) << "\n";


  // row col assignment test
  MatrixXd twoVec(2, 3);
  std::cout << twoVec.rows() << "\n";
  std::cout << twoVec.cols() << "\n";
  Vector3d vrow {5, 6, 7};
  Vector2d vcol {8, 9};
  twoVec.row(0) = vrow;
  std::cout << twoVec << "\n";
  twoVec.col(2) = vcol;
  std::cout << twoVec << "\n";

  Vector3d vgetrow = twoVec.row(0);
  std::cout << vgetrow << "\n";

  // Matrix initalization test
  MatrixXd vec1{ {1., 2., 3.} };
  std::cout << vec1.rows() << "\n";
  std::cout << vec1.cols() << "\n";

  // Array indexing test
  Array3d arr{5., 10., 15.};
  std::cout << arr[0] << "\n";
  std::cout << arr[2] << "\n";

  std::cout << arr.matrix() << "\nrow";
  std::cout << arr.matrix().rows() << "\n";
  std::cout << arr.matrix().cols() << "\n";
  std::cout << arr[0,0] << "\n";


}