#include "container/field.h"
#include "container/grid_curvilinear.h"
#include "container/grid_rectlinear.h"
#include "container/grid_cartesian.h"
#include "container/solution.h"
#include "common/io.h"
// #include "common/numerical.h"
// #include "container/grid_curvilinear.h"


#include <string>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <netcdf>
#include <cmath>
#include <chrono>
#include <type_traits>

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
  // Numerical Tests
  // Array3d a1{1., 2., 3.};
  // Array3d a2{1., 2., 5.};
  // // abs(-5);
  // // abs(a1-a2);

  // // std::cout << abs(-5) << "\n";
  // // std::cout << abs(a1-a2) << "\n";
  // // std::cout << isClose(a1, a2) << "\n";
  // // isClose(a1, a2);
  // // std::cout << isClose(1., 1.000000000001) << "\n";
  // std::cout << trigApprox(std::sin, M_PI*3/2) << "\n";
  // std::cout << std::sin((float)M_PI*3/2) << "\n";

  std::string vecpath = "data/mantle_phys.vec";
  std::string gridpath = "data/mantle_comp.grid";

  Solution<Vector3f> physSol = readVec(vecpath);
  // std::cout << physSol.strides[0] << " " << physSol.strides[1] << " " << physSol.strides[2] << "\n";
  // std::cout << physSol.getVal(20, 12, 88) << "\n";

  std::vector<std::vector<float>> coords = readGrid(gridpath);

  RectlinearGrid compGrid(&coords[0], &coords[1], &coords[2]);
  CurvilinearGrid curvi(physSol.getDimLen(0), physSol.getDimLen(1), physSol.getDimLen(2), physSol.getData());
  curvi.setCompGrid(&compGrid);
  std::vector<Array3f> phys = curvi.getVoxelCoords(100, 100, 100);
  std::vector<Array3f> comp = curvi.compGrid->getVoxelCoords(100, 100, 100);
  Array3f low_vtx = comp[0];
  Array3f high_vtx = comp[7];

  Array3f compGT = low_vtx + (high_vtx - low_vtx)*0.2;
  Array3f physQuery = sph2car(compGT(0),  radians(compGT(1)), radians(compGT(2)));
  std::cout << "Phys Query:\n" <<physQuery << "\n";
  std::cout << "GT comp:\n" <<  compGT << "\n";


  auto start = std::chrono::high_resolution_clock::now();
  Array3f compEst;
  int num_iter = 2000;
  for (int i = 0; i < num_iter; i++ ) {
    compEst = curvi.phys2comp_newtwon(physQuery, low_vtx, high_vtx, phys, 50);
  }
  std::cout << "Newton estimated comp: \n" <<compEst << "\n";
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
  std::cout << "Time taken in seconds: " << (double)duration.count()*0.001 << "\n";


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