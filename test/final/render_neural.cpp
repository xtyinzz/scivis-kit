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
  vtkStructuredGrid *sg = sgr->GetOutput();
  int *dims = sg->GetDimensions();
  double *dimBounds = sg->GetBounds();
  double tmpmin = dimBounds[0], tmpmax = dimBounds[1];
  dimBounds[0] = dimBounds[4];
  dimBounds[1] = dimBounds[5];
  dimBounds[4] = tmpmin;
  dimBounds[5] = tmpmax;
  std::cout << dimBounds[0] << " " << dimBounds[1] << " " << dimBounds[2] << " " <<
            dimBounds[3] << " " << dimBounds[4] << " " << dimBounds[5] << " \n";

  // build grid
  CartesianGrid cg(
    -1, 1, dims[2],
    -1, 1, dims[1],
    -1, 1, dims[0]
  );

  // build solution
  vtkPointData *pd = sg->GetPointData();

  std::cout << pd->GetNumberOfArrays() << "\n";
  vtkDataArray *thetaVTK = pd->GetArray(0);
  std::cout <<  "Number of tuples in theta:" << thetaVTK->GetNumberOfTuples() << "\n";
  std::vector<float> thetaData(thetaVTK->GetNumberOfTuples());
  for (int i = 0; i < thetaData.size(); i++) {
    thetaData[i] = (float) thetaVTK->GetTuple(i)[0];
  }
  std::cout << sizeof(thetaVTK->GetTuple(2222)[0]) << "\n";
  // thetaVTK->GetDataType()
  Solution<float> thetaSol(3);
  thetaSol.setDimLen(0, dims[2]);
  thetaSol.setDimLen(1, dims[1]);
  thetaSol.setDimLen(2, dims[0]);
  thetaSol.setData(thetaData);
  // get scalar field
  ScalarField sf(&cg, &thetaSol);
  std::cout <<  "Number of tuples in theta:" << thetaVTK->GetNumberOfTuples() << "\n";

  // *************** VR ****************
  
  std::vector<float> minmaxTheta = sf.getMinMax();
  float minTheta = minmaxTheta[0];
  float maxTheta = minmaxTheta[1];

  TransferFunction tf;
  tf.addRGBPoint(-std::numeric_limits<float>::max()/10,    0,    0,      0);
  tf.addRGBPoint(minTheta-1,0,0,0);
  tf.addRGBPoint(minTheta, 0.74509803921568629, 0.74509803921568629, 0.74509803921568629);
  tf.addRGBPoint(396.72797085061336, .99, 0. ,0.);
  tf.addRGBPoint(maxTheta+1, 0.7058820, 0.0156863,	0.1490200);
  tf.addRGBPoint(std::numeric_limits<float>::max()/10, 0, 0,	0);

  tf.addOpacityPoint(-std::numeric_limits<float>::max()/10, 0);
  tf.addOpacityPoint(0, 0.);
  tf.addOpacityPoint(minTheta, 0.);
  tf.addOpacityPoint(344.908, 0.552);
  tf.addOpacityPoint(369.881, 0.966);
  tf.addOpacityPoint(maxTheta, 0.966);
  tf.addOpacityPoint(maxTheta+0.5, 0);
  tf.addOpacityPoint(std::numeric_limits<float>::max()/10, 0);
    std::cout <<  "1Number of tuples in theta:" << thetaVTK->GetNumberOfTuples() << "\n";

  VolumeRenderer vr;

  vr.setField(&sf);
  vr.setTransferFunction(&tf);
  int imgWidth = std::atoi(argv[1]);
  vr.setImageDimension(imgWidth, imgWidth);

  // std::vector<std::string> networks = { "res", "attn", "dense", "siren" };
  std::vector<std::string> networks = { "res", "siren" };
  for (const std::string &network : networks) {
    std::cout <<  "Number of tuples in theta:" << thetaVTK->GetNumberOfTuples() << "\n";
    std::string networkPath = "pytorch/" + network + ".pt";
    NeuralCurvilinearGrid neuralCG(networkPath);
    std::cout <<  "Number of tuples in theta:" << thetaVTK->GetNumberOfTuples() << "\n";

    std::vector<std::vector<int>> planeIndices{
      {0, 1},
      {1, 2},
      {0, 2}
    };

    for (const std::vector<int> &planeIdx : planeIndices) {
      std::cout << "Rendering from plane " << planeIdx[0] << " and " << planeIdx[1] << "...\n";
      // depthPlaneIdx = (0+1+2) - two indices
      // int depthPlaneIdx = 3 - planeIdx[0] - planeIdx[1];
      // float depth = dimBounds[2*depthPlaneIdx + 1] - dimBounds[2*depthPlaneIdx];

      // glm::vec3 rayStep(0.f);
      // // float steplen = 0.4;
      // // int numSteps = (int) (depth / steplen + 0.5);
      // // rayStep[depthPlaneIdx] = steplen;

      int numSteps = imgWidth*2;
      // float steplen = depth / numSteps;
      // rayStep[depthPlaneIdx] = steplen;
      // numSteps = std::atoi(argv[2]);

      // std::cout << "W/H/Depth: " << imgWidth << "/" << imgWidth << "/" << numSteps << "\n";
    std::cout <<  "Number of tuples in theta:" << thetaVTK->GetNumberOfTuples() << "\n";
      std::vector<std::vector<glm::vec3>> physRays = vr.getRays(numSteps, planeIdx, dimBounds);
    std::cout <<  "Number of tuples in theta:" << thetaVTK->GetNumberOfTuples() << "\n";
      std::vector<std::vector<glm::vec3>> compRays = neuralCG.precomputeRayCompCoords(physRays);
    std::cout <<  "Number of tuples in theta:" << thetaVTK->GetNumberOfTuples() << "\n";

      vr.render(compRays, numSteps);
      std::string imageName = "nr_img/" + network + "_" + std::to_string(planeIdx[0]) + std::to_string(planeIdx[1]) + 
                              "_" + std::to_string(imgWidth) + "x2_" + 
                              std::to_string(numSteps) + ".png";
      vr.writePNG(imageName, 3);
    }
  }
}