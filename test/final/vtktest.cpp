#include "container/field.h"
#include "container/grid_curvilinear.h"
#include "container/grid_rectlinear.h"
#include "container/grid_cartesian.h"
#include "container/solution.h"

#include <string>
#include <vector>
#include <iostream>

#include <vtkStructuredGrid.h>
#include <glm/glm.hpp>
#include <Eigen/Dense>
using namespace Eigen;


int main() {
  vtkNew<vtkStructuredGrid> sg;
  double x[3] = {0., 0., 0.};
  vtkNew<vtkGenericCell> tmpGCell;
  vtkNew<vtkCell> cell = sg->FindCell(
    x,
    nullptr,
    nullptr,
    1e-4,

  );
}