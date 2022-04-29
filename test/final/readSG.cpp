#include "container/field.h"
#include "container/grid_curvilinear.h"
#include "container/grid_base.h"
#include "container/grid_rectlinear.h"
#include "container/grid_cartesian.h"
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


int main() {
  vtkNew<vtkXMLStructuredGridReader> sgr;
  sgr->SetFileName("data/bc80-45000-down.vts");
  sgr->Update();

  vtkStructuredGrid *sg = sgr->GetOutput();
  std::cout << sg->GetNumberOfPoints() << "\n";
  CurvilinearGrid curvGrid;
  curvGrid.setVTKStructuredGrid(sg);

  // std::cout << curvGrid.bounds[0] << " " << curvGrid.bounds[1] << "\n";
  // ************************** test getVoxelLerp
  // // std::cout << curvGrid.physGrid->GetNumberOfPoints() << "\n";
  CellLerp tmpcl;
  // // tmpcl = curvGrid.getVoxelLerp(0, 0, 0);
  // // std::cout << tmpcl.indices[0] << " " << tmpcl.indices[1] << " " << tmpcl.indices[2] << "\n";
  float query[3] = {10.0, -100.0, 192};
  tmpcl = curvGrid.getVoxelLerp(query[0], query[1], query[2]);
  // std::cout << tmpcl.indices[0] << " " << tmpcl.indices[1] << " " << tmpcl.indices[2] << "\n";
  std::cout << tmpcl.weights[0] << " " << tmpcl.weights[1] << " " << tmpcl.weights[2] << "\n";

  vtkPointData *pd = sg->GetPointData();
  vtkDataArray *thetaVTK = pd->GetArray(0);
  std::vector<float> thetaData(thetaVTK->GetNumberOfTuples());
  std::cout << "size is " << thetaData.size() << "\n";
  for (int i = 0; i < thetaData.size(); i++) {
    thetaData[i] = (float) thetaVTK->GetTuple(i)[0];
  }
  Solution<float> thetaSol(3);
  thetaSol.setDimLen(0, curvGrid.physGridDim[2]);
  thetaSol.setDimLen(1, curvGrid.physGridDim[1]);
  thetaSol.setDimLen(2, curvGrid.physGridDim[0]);
  thetaSol.setData(thetaData);
  ScalarField sf(&curvGrid, &thetaSol);





}