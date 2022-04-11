#include "container/field.h"
#include "container/grid_cartesian.h"
#include "container/solution.h"
#include "common/numerical.h"
#include "common/io.h"

#include <string>

#include <Eigen/Dense>
using namespace Eigen;

const int MAX_STEPS = 200;

int main() {
  // Matrix3Xf m = Matrix3Xf::Random(3,3);
  // Matrix3f mc = m;
  // // m[0] = MatrixX3f{1.f, 1.f, 1.f};
  // std::cout << m << "\n***";
  // std::cout << mc << "\n***";
  // std::cout << m*2 + m*3 << "\n***";
  // std::cout << m(2) << "\n";
  // std::cout << m.row(2) << "\n";

  // // std::cout << m(4) << "\n";
  // // std::cout << m(5) << "\n";


  std::string vecFilePath = "data/tornadoPC_96.vec";
  Solution<Array3f> vecData;
  vecData.fromVec(vecFilePath, true);
  // swap the value at 0, 1 dim of each vector in a vector solution
  vecData.swapVecDim(1, 0, 2);
  vecData.save("data/task4Vec.raw");
  // change data order to ZYX
  // vecData.swapAxes(2, 1, 0);

}