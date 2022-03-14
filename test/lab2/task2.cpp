#include "container/field.h"
#include "renderer/VolumeRenderer.h"
#include "renderer/PieceWiseFunction.h"
#include "renderer/TransferFunction.h"
#include "common.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

// #include "Eigen/Dense"

#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>



// using namespace Eigen;

// using namespace glm;

// args: plane_index0 plane_index1 imgDimension NumSteps

int main(int argc, char *argv[]) {
  TransferFunction tf;
  tf.addRGBPoint(0, 0.,0.,0.);
  tf.addRGBPoint(2000, 1., .7137, 0.75686); // light pink
  tf.addRGBPoint(2500, 1., 0., 0.); // red
  tf.addRGBPoint(3000, 0., 0.4, 0.); // dark green
  tf.addRGBPoint(5000, .698, .133, .133); // firebrick

  glm::vec4 opacities(0.002, 0.002, 0.4, 0.8);
  int numSteps = std::atoi(argv[4]);
  if (numSteps >= 512) {
    opacities /= 2.;
  }

  tf.addOpacityPoint(0, 0.);
  tf.addOpacityPoint(2000, opacities[0]);
  tf.addOpacityPoint(2500, opacities[1]);
  tf.addOpacityPoint(3000, opacities[2]);
  tf.addOpacityPoint(5000, opacities[3]);

  VolumeRenderer vr;

  std::string pdata = "../lab1/data/raw/resampled_256^3.raw";
  Solution<float> solution(256, 256, 256);
  solution.load(pdata);
  Grid g(256, 256, 256);
  Field<float> field(&g, &solution);

  vr.setField(&field);
  vr.setTransferFunction(&tf);
  int imgWidth = std::atoi(argv[3]);
  vr.setImageDimension(imgWidth, imgWidth);


  
  int planeIdx[2] = {std::atoi(argv[1]), std::atoi(argv[2])};
  // depthPlaneIdx = (0+1+2) - two indices
  int depthPlaneIdx = 3 - planeIdx[0] - planeIdx[1];
  int depth = field.getDimLen(depthPlaneIdx);
  glm::vec3 rayStep(0.f);
  float steplen = g.getDimSpacing(depthPlaneIdx)/2.5f;
  rayStep[depthPlaneIdx] = steplen;

  int useShading = std::atoi(argv[5]);
  std::string nameShading = "";
  if (useShading) {
    field.computeGradSolution3D();
    // field.grad->save("data/tmpgrad.raw");
    std::cout << field.grad->getDimLen(0) << " " << field.grad->getDimLen(1) << " " << field.grad->getDimLen(2) << "\n";
    // vr.setGradField(field.grad);
    vr.setShading(vr.SHADING_PHONG);
    glm::vec3 lightPos(256.f, 256.f, 0.f);
    vr.setLight(lightPos, 1., 1., 1.);
    nameShading = "_PHONG";
  }

  // std::cout << vr.gradField->getDimLen(0) << " " << vr.gradField->getDimLen(1) << " " << vr.gradField->getDimLen(2) << "\n";
  // printVec(vr.gradField->getVal(0.,0.,0.));
  // printVec(vr.gradField->getVal(0.8,0.,0.));

  vr.render(rayStep, numSteps, planeIdx);;
  std::string imageName = "img/vr_" + std::to_string(planeIdx[0]) + std::to_string(planeIdx[1]) + 
                          "_" + std::to_string(imgWidth) + "x2_" + 
                          "256of" + std::to_string(steplen) + "step" + 
                          nameShading + ".png";
  vr.writePNG(imageName, 3);

}