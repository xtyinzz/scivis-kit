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
  VolumeRenderer vr;

  TransferFunction tf;
  tf.setDefaultMinMaxRGB(glm::vec3(0.9,0.1, 0.1), glm::vec3(0.9,0.1, 0.1));
  tf.addRGBPoint(-std::numeric_limits<float>::max()/10, .9f, .1f, .1f);
  tf.addRGBPoint(std::numeric_limits<float>::max()/10, .9f, .1f, .1f);
  // std::cout << tf.getRGBPointRange()[0] << tf.getRGBPointRange()[1] << "\n";

  tf.setDefaultMinMaxOpacity(0.8, 0);
  tf.addOpacityPoint(0.0000001, 0.8);
  tf.addOpacityPoint(0.0000002, 0);

  int dimLen = std::atoi(argv[3]);
  float xmin = -2;
  float xmax = 2;
  float ymin = -2;
  float ymax = 2;
  float zmin = -2;
  float zmax = 2;
  float dx = (xmax - xmin) / (dimLen - 1);
  float dy = (xmax - xmin) / (dimLen - 1);
  float dz = (xmax - xmin) / (dimLen - 1);

  Grid g(xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz);
  Solution<float> dummyS;
  HeartEquation<float> fieldHE(1.f, 1.f);
  fieldHE.setGrid(&g);
  fieldHE.setSolution(&dummyS);

  vr.setField(&fieldHE);
  vr.setTransferFunction(&tf);
  int imgWidth = std::atoi(argv[3]);
  vr.setImageDimension(imgWidth, imgWidth);

  int numSteps = std::atoi(argv[4]);
  int planeIdx[2] = { std::atoi(argv[1]), std::atoi(argv[2]) };
  // depthPlaneIdx = (0+1+2) - two indices
  int depthPlaneIdx = 3 - planeIdx[0] - planeIdx[1];
  // int depth = imgWidth;
  glm::vec3 rayStep(0.f);
  float steplen = 4./imgWidth;
  rayStep[depthPlaneIdx] = steplen;

  int useShading = std::atoi(argv[5]);
  std::string nameShading = "";
  if (useShading) {
    // field.computeGradSolution3D();
    // field.grad->save("data/tmpgrad.raw");
    // std::cout << field.grad->getDimLen(0) << " " << field.grad->getDimLen(1) << " " << field.grad->getDimLen(2) << "\n";

    // vr.setGradField(field.grad);
    vr.setShading(vr.SHADING_PHONG);
    glm::vec3 lightPos(-2.f, -2.f, 1.f);
    vr.setLight(lightPos, .1f, .8f, .2f);
    nameShading = "_PHONG";
  }

  vr.render(rayStep, numSteps, planeIdx);
  std::string imageName = "img/heart_" + std::to_string(planeIdx[0]) + std::to_string(planeIdx[1]) + 
                          "_" + std::to_string(imgWidth) + "x2_" + 
                          std::to_string(numSteps) + "of" + std::to_string(steplen) + "step" + 
                          nameShading + ".png";
  vr.writePNG(imageName, 3);

}