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
  // std::string vecFilePath = "data/10.vec";
  Solution<Array3f> vecData;
  vecData.fromVec(vecFilePath, true);
  // swap the value at 0, 1 dim of each vector in a vector solution
  vecData.swapVecDim(1, 0, 2);
  // change data order to ZYX
  vecData.swapAxes(2, 1, 0);
  
  CartesianGrid vecGrid(vecData.getDimLen(0), vecData.getDimLen(1), vecData.getDimLen(2));
  VectorField<Array3f> vecField(&vecGrid, &vecData);
  // //////////////////// PARTICLE TRACING

  std::string seedFile = "data/seeds.txt";
  std::vector<Vector3f> seeds = readSeeds(seedFile);
  {
    // task 1
    std::vector<std::vector<Array3f>> traces;
    for (const Vector3f seed : seeds) {
      // std::cout << seed << "\n***\n";
      Array3f seedArray = seed.array();
      // seedArray /= 95;
      std::vector<Array3f> streamline = particleTracingRK1(seedArray, &vecField, 8, MAX_STEPS);
      traces.push_back(streamline);
    }
    writeParticleTrace("data/sl_rk1.raw", traces);
    writeParticleTraceLength("data/sl_len_rk1.raw", traces);
  }

  {
    // task 2
    std::vector<std::vector<Array3f>> traces;
    for (const Vector3f seed : seeds) {
      // std::cout << seed << "\n***\n";
      Array3f seedArray = seed.array();
      // seedArray /= 95;
      std::vector<Array3f> streamline = particleTracingRK4(seedArray, &vecField, 8, MAX_STEPS);
      traces.push_back(streamline);
    }
    writeParticleTrace("data/sl_rk4.raw", traces);
    writeParticleTraceLength("data/sl_len_rk4.raw", traces);
  }
}