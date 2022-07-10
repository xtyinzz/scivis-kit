#include <container/field.h>
#include <container/grid_curvilinear.h>
#include <container/grid_cartesian.h>

#include <torch/script.h>
#include <vtkStructuredGrid.h>
#include <vtkStaticCellLocator.h>
#include <vtkPointData.h>
#include <vtkXMLStructuredGridReader.h>
#include <vtkCellLocator.h>
#include <vtkFloatArray.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <Eigen/Dense>

#include <omp.h>
#include <vector>

#include <chrono>

std::vector<float> flattenVectorGLM(std::vector<std::vector<glm::vec3>> physRays) {
  std::vector<float> raysSTL;
  for (int i = 0; i < physRays.size(); i++) {
    std::vector<float> raySTL(physRays[i].size()*3);
    // flatten a vector of vec3
    for (int j = 0; j < physRays[i].size(); j++) {
      raySTL[j*3] = physRays[i][j].x;
      raySTL[j*3+1] = physRays[i][j].y;
      raySTL[j*3+2] = physRays[i][j].z;
    }
    raysSTL.insert(std::end(raysSTL), std::begin(raySTL), std::end(raySTL));
  }
  return raysSTL;
}



int main(int argc, char *argv[]) {

  glm::vec3 glmvec(1);
  glm::vec3 glmvec2(2);
  glm::vec3 glmvec3(3);
  std::vector<std::vector<float>> stlnested;
  std::vector<glm::vec3> stlglm;
  stlglm.push_back(glmvec);
  stlglm.push_back(glmvec2);
  stlglm.push_back(glmvec3);
  std::vector<std::vector<glm::vec3>> stlglmnested;
  stlglmnested.push_back(stlglm);
  stlglmnested.push_back(stlglm);
  std::vector<float> stl = flattenVectorGLM(stlglmnested);

  // std::vector<float> stl(&glmvec.x, &glmvec.x + 3);
  // stlnested.push_back(stl);
  // stlnested.push_back(stl);
  torch::Tensor test = torch::tensor(stl);
  std::cout<< test.reshape({-1, 3}) << "\n";

  // torch::jit::script::Module module;
  // torch::jit::script::Module module1;
  // module = torch::jit::load("pytorch/traced_2499.pt");
  // module1 = torch::jit::load("pytorch/traced_2499_1.pt");
  // int num = std::atoi(argv[1]);
  // torch::Tensor example = torch::rand({num*num*num, 3});
  // torch::Tensor example1 = torch::rand({1, 3});

  // {
  //   // network inference time
  //   auto start = std::chrono::high_resolution_clock::now();
  //   std::vector<torch::jit::IValue> inputs;
  //   inputs.push_back(example);
  //   module1.forward(inputs);

  //   inputs[0] = example1;
  //   module1.forward(inputs);

  //   // for (int i = 0; i < 2000; i++) {
  //   //   module.forward(inputs);
  //   // }
  //   auto end = std::chrono::high_resolution_clock::now();
  //   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
  //   std::cout << "Time taken in seconds: " << (double)duration.count()*0.001 << "\n";
  // }

  // // std::cout << module.forward(inputs) << "\n\n";
  // // std::cout << module.forward(inputs).toTensor() << "\n";

  // // chimera time

  // vtkNew<vtkXMLStructuredGridReader> sgr;
  // sgr->SetFileName("data/bc80-45000-down.vts");
  // sgr->Update();

  // vtkStructuredGrid *sg = sgr->GetOutput();
  // std::cout << sg->GetNumberOfPoints() << "\n";
  // CurvilinearGrid curvGrid;
  // curvGrid.setVTKStructuredGrid(sg);

  // // std::cout << curvGrid.bounds[0] << " " << curvGrid.bounds[1] << "\n";
  // // ************************** test getVoxelLerp
  // // // std::cout << curvGrid.physGrid->GetNumberOfPoints() << "\n";
  // CellLerp tmpcl;
  // // // tmpcl = curvGrid.getVoxelLerp(0, 0, 0);
  // // // std::cout << tmpcl.indices[0] << " " << tmpcl.indices[1] << " " << tmpcl.indices[2] << "\n";
  // float query[3] = {300, 100.0, 10};

  // {
  //   auto start = std::chrono::high_resolution_clock::now();
  //   #pragma omp parallel for
  //   for (int i = 0; i < num*num*num; i++) {
  //     tmpcl = curvGrid.getVoxelLerp(query[0], query[1], query[2]);
  //   }
  //   auto end = std::chrono::high_resolution_clock::now();
  //   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
  //   std::cout << "Time taken in seconds: " << (double)duration.count()*0.001 << "\n";
  // }

}