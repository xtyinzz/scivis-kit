#include "container/field.h"
#include "container/grid_curvilinear.h"
#include "renderer/TransferFunction.h"
#include "renderer/VolumeRenderer.h"
#include "container/solution.h"
#include "common/io.h"

#include <string>
#include <vector>
#include <iostream>
#include <chrono>

#include <vtkStructuredGrid.h>
#include <vtkStaticCellLocator.h>
#include <vtkPointData.h>
#include <vtkXMLStructuredGridReader.h>
#include <vtkCellLocator.h>
#include <vtkFloatArray.h>
#include <glm/glm.hpp>
#include <Eigen/Dense>
using namespace Eigen;


int main(int argc, char *argv[]) {
  // *************** Data ****************
  vtkNew<vtkXMLStructuredGridReader> sgr;
  sgr->SetFileName("data/bc80-45000-down.vts");
  sgr->Update();

  // build grid
  vtkStructuredGrid *sg = sgr->GetOutput();
  std::cout << sg->GetNumberOfPoints() << "\n";
  CurvilinearGrid curvGrid;
  curvGrid.setVTKStructuredGrid(sg);

  // build solution
  vtkPointData *pd = sg->GetPointData();
  vtkDataArray *thetaVTK = pd->GetArray(0);
  std::vector<float> thetaData(thetaVTK->GetNumberOfTuples());
  for (int i = 0; i < thetaData.size(); i++) {
    thetaData[i] = (float) thetaVTK->GetTuple(i)[0];
  }
  Solution<float> thetaSol(3);
  thetaSol.setDimLen(0, curvGrid.physGridDim[2]);
  thetaSol.setDimLen(1, curvGrid.physGridDim[1]);
  thetaSol.setDimLen(2, curvGrid.physGridDim[0]);
  thetaSol.setData(thetaData);
  // get scalar field
  ScalarField sf(&curvGrid, &thetaSol);

  // *************** VR ****************
  
  std::vector<float> minmaxTheta = sf.getMinMax();
  float minTheta = minmaxTheta[0];
  float maxTheta = minmaxTheta[1];

  TransferFunction tf;
  tf.addRGBPoint(-std::numeric_limits<float>::max()/10,    0,    0,      0);
  tf.addRGBPoint(minTheta-1,    0,0,0);
  tf.addRGBPoint(minTheta,   0.74509803921568629, 0.74509803921568629, 0.74509803921568629);
  tf.addRGBPoint(396.72797085061336, .99, 0. ,0.);
  tf.addRGBPoint(maxTheta+1, 0.7058820, 0.0156863,	0.1490200);
  tf.addRGBPoint(std::numeric_limits<float>::max()/10, 0, 0,	0);

  tf.addRGBPoint(-std::numeric_limits<float>::max()/10,    0, 0, 0);
  tf.addOpacityPoint(0, 0.);
  tf.addOpacityPoint(minTheta, 0.);
  tf.addOpacityPoint(344.908, 0.552);
  tf.addOpacityPoint(369.881, 0.966);
  tf.addOpacityPoint(maxTheta, 0.966);
  tf.addOpacityPoint(maxTheta+0.5, 0);
  tf.addOpacityPoint(std::numeric_limits<float>::max()/10, 0);

  VolumeRenderer vr;

  vr.setField(&sf);
  vr.setTransferFunction(&tf);
  int imgWidth = std::atoi(argv[1]);
  vr.setImageDimension(imgWidth, imgWidth);

  std::vector<std::vector<int>> planeIndices{
    {0, 1},
    {1, 2},
    {0, 2}
  };

  for (const std::vector<int> &planeIdx : planeIndices) {
    std::cout << "Rendering from plane " << planeIdx[0] << " and " << planeIdx[1] << "...\n";
    // depthPlaneIdx = (0+1+2) - two indices
    int depthPlaneIdx = 3 - planeIdx[0] - planeIdx[1];
    int depth = sf.getDimExtent(0)[1] - sf.getDimExtent(0)[0];

    glm::vec3 rayStep(0.f);
    float steplen = 1;
    int numSteps = (int) (depth / steplen + 0.5);
    rayStep[depthPlaneIdx] = steplen;

    // int numSteps = imgWidth*2;
    // float steplen = depth / numSteps;
    // rayStep[depthPlaneIdx] = steplen;
    // numSteps = std::atoi(argv[2]);

    vr.render(rayStep, numSteps, planeIdx);
    std::string imageName = "img/bc80-450000-thetaVR_" + std::to_string(planeIdx[0]) + std::to_string(planeIdx[1]) + 
                            "_" + std::to_string(imgWidth) + "x2_" + 
                            std::to_string(numSteps) + "of" + std::to_string(steplen) + "step" + ".png";
    vr.writePNG(imageName, 3);
  }
}