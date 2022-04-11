#include "container/field.h"
#include "container/grid_cartesian.h"
#include "container/solution.h"
#include "renderer/TransferFunction.h"
#include "renderer/VolumeRenderer.h"
#include "common/numerical.h"
#include "common/io.h"

#include <string>

#include <Eigen/Dense>
#include <glm/glm.hpp>
using namespace Eigen;

const int MAX_STEPS = 200;

int main(int argc, char *argv[]) {
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


  //////////////////// VOTORCITY
  vecField.computeJac3D();
  vecField.computeVort3D();
  vecField.computeVortMag();
  std::cout << "Vorticity Magnitude scalar field computation completed\n";

  //////////////////// Volume Rendering
  ScalarField<float> vortMagField(&vecGrid, vecField.vortMagSol);


  //////// TF
  std::vector<float> minmaxVortMag = vortMagField.getMinMax();
  float midMag = (minmaxVortMag[0]+minmaxVortMag[1])/2.f;
  TransferFunction tf;
  tf.addRGBPoint(minmaxVortMag[0],    .2314,    .298,      .753);
  tf.addRGBPoint(midMag,    1.  ,    1.,      1.);
  tf.addRGBPoint(minmaxVortMag[1], .7059,  .0157,    .149);
  Array3f opacities(0., 0.052, 1.);
  float oneQuarterMag = (minmaxVortMag[0]+midMag)/2.f;
  tf.addOpacityPoint(minmaxVortMag[0], 0.);
  tf.addOpacityPoint(oneQuarterMag, opacities[1]);
  tf.addOpacityPoint(minmaxVortMag[1], opacities[2]);

  ///////// VR
  VolumeRenderer vr;

  vr.setField(&vortMagField);
  vr.setTransferFunction(&tf);
  int imgWidth = std::atoi(argv[1]);
  vr.setImageDimension(imgWidth, imgWidth);
  
  std::vector<std::vector<int>> planeIndices{
    {0, 1}, 
    {1, 2},
    {0, 2}
  };


  int useShading = std::atoi(argv[3]);
  std::string nameShading = "";
  if (useShading) {
    vortMagField.computeGradSolution3D();
    vr.setShading(vr.SHADING_PHONG);
    glm::vec3 lightPos(96.f, 96.f, 0.f);
    vr.setLight(lightPos, 0.25, 0.5, 0.5);
    nameShading = "_PHONG";
  }

  for (const std::vector<int> &planeIdx : planeIndices) {
    std::cout << "Rendering from plane " << planeIdx[0] << " and " << planeIdx[1] << "...\n";
    // depthPlaneIdx = (0+1+2) - two indices
    int depthPlaneIdx = 3 - planeIdx[0] - planeIdx[1];
    int depth = vortMagField.getDimLen(depthPlaneIdx);
    glm::vec3 rayStep(0.f);
    float steplen = vecGrid.getDimSpacing(depthPlaneIdx)/2.5f;
    rayStep[depthPlaneIdx] = steplen;
    int numSteps = (int) (vecGrid.getDimLen(depthPlaneIdx) / steplen + 0.5);
    numSteps = std::atoi(argv[2]);

    vr.render(rayStep, numSteps, planeIdx);
    std::string imageName = "img/vortMagVR_" + std::to_string(planeIdx[0]) + std::to_string(planeIdx[1]) + 
                            "_" + std::to_string(imgWidth) + "x2_" + 
                            std::to_string(numSteps) + "of" + std::to_string(steplen) + "step" + 
                            nameShading + ".png";
    vr.writePNG(imageName, 3);
  }




}