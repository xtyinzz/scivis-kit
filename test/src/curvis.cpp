#include "container/field.h"
#include "container/grid_curvilinear.h"
#include "container/grid_rectlinear.h"
#include "container/grid_cartesian.h"
#include "container/solution.h"
#include "common/io.h"
// #include "container/grid_curvilinear.h"


#include <string>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <netcdf>

using namespace Eigen;

void printTmp(std::vector<int> *v) {
  std::vector<int> myv(*v);
  std::cout << myv[0] << "!!!\n";
}

void eigenType(VectorXf m) {
  std::cout << m << "\n";
}

// using Eigen::MatrixXd;
int main() {
  // std::string vecpath = "data/mantle_phys.vec";
  // std::string gridpath = "data/mantle_comp.grid";

  // Solution<Vector3f> vecSol = readVec(vecpath);
  // std::cout << vecSol.strides[0] << " " << vecSol.strides[1] << " " << vecSol.strides[2] << "\n";
  // std::cout << vecSol.getVal(20, 12, 88) << "\n";

  // std::vector<VectorXf> coords = readGrid(gridpath);

  VectorXf vf(3);
  vf << 1.f, 2.f, 3.f;
  vf(0) = 9;
  vf(vf.size()-1) = 22;
  eigenType(vf);

  // writeVec("data/tmpphys.vec", vecSol.getDimLen(0), vecSol.getDimLen(1), vecSol.getDimLen(2), vecSol.getData());
  // readGrid("data/mantle_comp.grid");
  // std::vector<int> v{1,2,3};
  // printTmp(&v);
  // std::vector<double> coord = {1., 2., 3.};
  // RectlinearGrid rg(&coord, &coord, &coord);
  // // std::cout << rg.getDomain(0)[0] << "\n";
  // // std::cout << rg.getDomain(1)[1] << "\n";
  // // std::cout << rg.getDomain(2)[1] << "\n";

  // Field<float> f(&rg);
  // std::cout << f.getDimExtent(0)[0] << "\n";
  // std::cout << f.getDimExtent(0)[1] << "\n";

}