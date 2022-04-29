#include "container/field.h"
#include "container/grid_curvilinear.h"
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
#include <vtkXMLStructuredGridReader.h>
#include <vtkCellLocator.h>
#include <vtkFloatArray.h>
#include <glm/glm.hpp>
#include <Eigen/Dense>
using namespace Eigen;


int main() {
  std::string vecpath = "data/mantle_phys.vec";
  Solution<Vector3f> physSol = readVec(vecpath);
  const int sgDim[3] = {physSol.getDimLen(0), physSol.getDimLen(1), physSol.getDimLen(2)};
  // std::cout << physSol.strides[0] << " " << physSol.strides[1] << " " << physSol.strides[2] << "\n";
  // std::cout << physSol.getVal(20, 12, 88) << "\n";

  CurvilinearGrid curvi(sgDim[0], sgDim[1], sgDim[2], physSol.getData());

  vtkNew<vtkPoints> points;
  std::vector<Vector3f> physData = physSol.getData();
  for (int k = 0; k < physSol.getDimLen(2); k++) {
    for (int j = 0; j < physSol.getDimLen(1); j++) {
      for (int i = 0; i < physSol.getDimLen(0); i++) {
        Vector3f coord = physSol.getVal(i, j, k);
        points->InsertNextPoint(coord.data());
      }
    }
  }
  for (size_t i = 0; i < physSol.length; i++) {
    points->InsertNextPoint(physData[i](0), physData[i](1), physData[i](2));
  }


  vtkNew<vtkXMLStructuredGridReader> sgReader;
  sgReader->SetFileName("data/bc80-45000-down.vts");
  sgReader->Update();
  
  vtkNew<vtkStructuredGrid> sg;
  // int sgDim[3] = {physSol.getDimLen(0), physSol.getDimLen(1), physSol.getDimLen(2)};
  sg->SetDimensions(sgDim);
  sg->SetPoints(points);
  std::cout << sg->GetNumberOfCells() << "\n";
  double x1[3] = {0., 0., 0.};
  double x2[3] = {754.9644 ,  4163.159  , -2526.5234};
  vtkNew<vtkGenericCell> tmpGCell;
  double pcoord[3];
  double weights[8];
  int subid;
  vtkIdType cid = sg->FindCell(
    x2,
    NULL,
    tmpGCell,
    0,
    0,
    subid,
    pcoord,
    weights
  );

  std::cout << cid << ":cell id\n";
  std::cout << tmpGCell->PointIds[0] << ":cell id\n";
  vtkPoints *cellPoints = tmpGCell->GetPoints();
  for (int i = 0; i < tmpGCell->GetNumberOfPoints(); i++) {
    double *coord = cellPoints->GetPoint(i);
    Vector3d cvec(coord);
    std::cout << i << "\n" << cvec << "\n\n";
  }
  // cid = sg->FindCell(
  //   x2,
  //   NULL,
  //   tmpGCell,
  //   0,
  //   1e-4,
  //   subid,
  //   pcoord,
  //   weights
  // );
  // std::cout << cid << ":cell id\n";
  // vtkNew<vtkStaticCellLocator> cloc;
  // vtkNew<vtkCellLocator> cloc;
  // cloc->SetDataSet(sg);
  // cloc->BuildLocator();
  // vtkIdType cid = cloc->FindCell(x1);
  // std::cout << cid << ":cell id\n";
  // cid = cloc->FindCell(x2);
  // std::cout << cid << ":cell id\n";

  // std::vector<Array3f> phys = curvi.getVoxelCoords(100, 100, 100);
  // std::vector<Array3f> comp = curvi.compGrid->getVoxelCoords(100, 100, 100);
  // Array3f low_vtx = comp[0];
  // Array3f high_vtx = comp[7];

  // Array3f compGT = low_vtx + (high_vtx - low_vtx)*0.2;
  // Array3f physQuery = sph2car(compGT(0),  radians(compGT(1)), radians(compGT(2)));
  // std::cout << "Phys Query:\n" << physQuery << "\n";
  // std::cout << "GT comp:\n" <<  compGT << "\n";


  // auto start = std::chrono::high_resolution_clock::now();
  // Array3f compEst;
  // int num_iter = 2000;
  // for (int i = 0; i < num_iter; i++ ) {
  //   compEst = curvi.phys2comp_newtwon(physQuery, low_vtx, high_vtx, phys, 50);
  // }
  // std::cout << "Newton estimated comp: \n" <<compEst << "\n";
  // auto end = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
  // std::cout << "Time taken in seconds: " << (double)duration.count()*0.001 << "\n";
}