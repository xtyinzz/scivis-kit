#ifndef SVK_CONTRAINER_GRID_CURVILINEAR_H
#define SVK_CONTRAINER_GRID_CURVILINEAR_H

// #include "common.h"
#include "grid_base.h"
#include "grid_rectlinear.h"
#include "solution.h"
#include "common/numerical.h"
#include "common/util.h"
#include <assert.h>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>

#include <torch/script.h>
#include <Eigen/Dense>
#include <vtkStructuredGrid.h>
#include <vtkStaticCellLocator.h>
#include <vtkCellLocator.h>
#include <vtkXMLStructuredGridReader.h>

using namespace Eigen;

class NewtonMethod {
  public:
    int maxiter=50;
    float atol=1.48e-8;
    float rtol=0.0;
};

// template <typename T>
class CurvilinearGrid: public GridBase {
  private:

  public:
    std::string gtype = "Curvilinear";
    NewtonMethod newton;
    std::vector<DimPropertyBase> dims;
    RectlinearGrid *compGrid;
    Solution<Vector3f> coords;
    Array<float, 3, 2> bbox;
    vtkStructuredGrid *physGrid;
    int iorder[8] = {0, 1, 1, 0,  0, 1, 1, 0};
    int jorder[8] = {0, 0, 1, 1,  0, 0, 1, 1};
    int korder[8] = {0, 0, 0, 0,  1, 1, 1, 1};
    float bounds[6];
    // vtkNew<vtkStructuredGrid> physGrid;
    int *physGridDim;
    int strides[3];
    // cell count, vertex count
    int ccount = 0;
    int vcount = 0;

    CurvilinearGrid() {}
    CurvilinearGrid(int numDims) {
      this->dims.resize(numDims);
    }
    CurvilinearGrid(int xdim, int ydim, int zdim) {
      this->dims.resize(3);
      this->coords = Solution<Vector3f>(xdim, ydim, zdim);
      // vcount = this->dims[0].len * this->dims[1].len * this->dims[2].len;
      // ccount = (this->dims[0].len - 1) * (this->dims[1].len - 1) * (this->dims[2].len - 1);
    }
    CurvilinearGrid(int xdim, int ydim, int zdim, const std::vector<Vector3f> &coords): CurvilinearGrid(xdim, ydim, zdim) {
      this->coords.setData(coords);
      // vcount = this->dims[0].len * this->dims[1].len * this->dims[2].len;
      // ccount = (this->dims[0].len - 1) * (this->dims[1].len - 1) * (this->dims[2].len - 1);
    }

    // void constructStructuedGrid(int dims[3], vtkPoints *points) {
    //   this->physGrid->SetDimensions(dims);
    //   this->physGrid->SetPoints(points);
    //   this->physGridDim = this->physGrid->GetDimensions();
    // }

    // void readStructuredGrid(const std::string &filename) {
    //   vtkNew<vtkXMLStructuredGridReader> sgr;
    //   sgr->SetFileName(filename.c_str());
    //   sgr->Update();
    //   this->physGrid = sgr->GetOutput();
    //   this->physGridDim = this->physGrid->GetDimensions();
    //   this->strides[0] = 1;
    //   this->strides[1] = this->physGridDim[0];
    //   this->strides[2] = this->physGridDim[0]*this->physGridDim[1];
    // }
    void setVTKStructuredGrid(vtkStructuredGrid *sg) {
      this->physGrid = sg;
      this->physGridDim = this->physGrid->GetDimensions();
      this->strides[0] = 1;
      this->strides[1] = this->physGridDim[0];
      this->strides[2] = this->physGridDim[0]*this->physGridDim[1];
      
      double *dbounds = this->physGrid->GetBounds();
      for (int i = 0; i < 6; i++) {
        this->bounds[i] = (float)dbounds[i];
      }
    }

    // TODO
    void updateCounts() {
      // vcount = this->dims[0].len * this->dims[1].len * this->dims[2].len;
      // ccount = (this->dims[0].len - 1) * (this->dims[1].len - 1) * (this->dims[2].len - 1);
    }

    int getCellCount() { return this->ccount; }
    int getVtxCount() { return this->vcount; }
    int getDimCount() { return this->dims.size(); } // what's return "dimensions of the grid" in hw write-up?

    std::vector<float> getDomain(int idim) {
      return std::vector<float>{bounds[2*idim], bounds[2*idim + 1]};
    }

    int getDimLen(int idim) { return this->dims[idim].len; }

    // ************************************************************************
    // core functions

    void setCompGrid(RectlinearGrid *compGrid) {
      this->compGrid = compGrid;
    }

    std::vector<int> vtkGetIJK(vtkIdType pid) {
      int k = pid / this->strides[2];
      pid -= k*this->strides[2];
      int j = pid / this->strides[1];
      pid -= j*this->strides[1];
      int i = (int)pid;
      // printf("%d %d %d\n", i, j, k);
      return std::vector<int> {i, j, k};
    }

    // phys2comp
    // Given phys coord, output cell indices and comp space lerp weights by Newton's Method
    // return floor corner x,y,z index and lerp weights on x,y,z
    virtual CellLerp getVoxelLerp(float x, float y, float z) {
      // assert(this->isBounded(x,y,z));
      // 1. locate the cell
      double physDouble[3] = {(double)x, (double)y, (double)z};
      // printf("%f\n", physDouble[0]);
      vtkNew<vtkGenericCell> tmpGCell;
      double pcoord[3];
      double vtkweights[8];
      int subid;
      this->physGrid->FindCell(physDouble, NULL, tmpGCell, 0, 1e-3, subid, pcoord, vtkweights);
      // if no cell found, return 0 weights, so getVal in field return 0 value
      if (tmpGCell->GetNumberOfPoints() == 0) {
        return CellLerp{std::vector<int>{0,0,0}, std::vector<float>{0., 0., 0.}};
      }
      // what's the cell get point heuristic?
      vtkIdList *pointIDs = tmpGCell->GetPointIds();
      vtkIdType cornerPID = pointIDs->GetId(0);
      // get voxel coords in vtk point order in cell 
      std::vector<Array3f> voxelCoords;
      voxelCoords.resize(8);
      for (int i = 0; i < 8; i++) {
        double *coord = this->physGrid->GetPoint(pointIDs->GetId(i));
        Array3f physf;
        physf[0] = (float)coord[0];
        physf[1] = (float)coord[1];
        physf[2] = (float)coord[2];
        voxelCoords[i] = physf;
        // std::cout << "\n" << pointIDs->GetId(i) << "\n";
      }
      // vtk cell point order different than scivis-kit assumption
      swapElement(voxelCoords, 2, 3);
      swapElement(voxelCoords, 6, 7);

      // double physDouble[3] = { (double)x, (double)y, (double)z };
      // double pcoord[3];
      // double vtkweights[8];
      // int subid;
      // vtkIdType cid = this->physGrid->FindCell(physDouble, NULL, 0, 0, subid, pcoord, vtkweights);
      // // if no cell found, return 0 weights, so getVal in field return 0 value
      // if (cid == -1) {
      //   return CellLerp{std::vector<int>{0, 0, 0}, std::vector<float>{0., 0., 0.}};
      // }
      // // get voxel coords in vtk point order in cell 
      // std::vector<Array3f> voxelCoords;
      // voxelCoords.resize(8);
      // for (int i = 0; i < 8; i++) {
      //   vtkIdType tmppid = cid + iorder[i]*strides[0] + jorder[i]*strides[1] + korder[i]*strides[2];
      //   double *coord = this->physGrid->GetPoint(tmppid);
      //   Array3f physf;
      //   physf[0] = (float)coord[0];
      //   physf[1] = (float)coord[1];
      //   physf[2] = (float)coord[2];
      //   voxelCoords[i] = physf;
      //   // std::cout << "\n" << physf << "\n";
      // }
      // vtk cell point order different than scivis-kit assumption
      // swapElement(voxelCoords, 2, 3);
      // swapElement(voxelCoords, 6, 7);



      // 2. find comp
      // newton's method
      Array3f phys{ x, y, z };
      Array3f comp = this->phys2comp_newtwon(
        phys, voxelCoords, this->newton.maxiter, this->newton.atol, this->newton.rtol
      );
      std::vector<float> weights = { comp(0), comp(1), comp(2) };
      std::vector<int> ijk = vtkGetIJK(cornerPID);
      // 3. trilinear interpolate in comp grid
      return CellLerp{ ijk, weights };
    }


    Array3f phys2comp_newtwon(const Array3f &phys, std::vector<Array3f> physVoxelCoord,
    int maxiter=10, float atol=1.48e-4, float rtol=0.0) {
      // Array3f init_comp = (lowVtx + highVtx) / 2.;
      Array3f comp{.5f, .5f, .5f};
      Array3f lowVtx{0.f, 0.f, 0.f};
      Array3f highVtx{1.f, 1.f, 1.f};
      // std::cout << init_comp << "\n";
      // Array3f goal_diff{0., 0., 0.};
      
      for(int i = 0; i < maxiter; i++) {
        std::vector<std::vector<Array3f>> coeff = trilerpSysEqCoeff(comp, lowVtx, highVtx, physVoxelCoord);

        // Array3f phys_est = trilerpSysEq(comp[0], comp[1], comp[2], coeff);
        Array3f phys_est = coeff[0][0];
        // std::cout << i << "th coeff: " << coeff[1][0] << "\n";
        // std::cout << i << "th: " << phys_est << "\n";

        Matrix3Xf diff_funcval = phys_est - phys;
        Matrix3f jac_inv = this->getJacInvComp2Phys(comp, coeff[1]);
        // std::cout << "My Inv \n" << jac_inv << "\n\n";
        Array3f new_comp = comp - (jac_inv * diff_funcval).array();
        // std::cout << "My Inv \n" << jac_inv <<  "  --  " << diff_funcval << "\n\n";
        // std::cout << "phys_est: \n" << phys_est << "\ncomp: \n" << new_comp << "\nerror: \n" << abs(diff_funcval.array()) << "\n";
        
        if (allClose(comp, new_comp, rtol, atol)) {
          return new_comp;
        }
        comp = new_comp;
      }
      return comp;
    }

    Matrix3f getJacComp2Phys(Array3f comp, std::vector<Array3f> coeff) {
      Matrix3f jac{
        {
          coeff[1][0] + coeff[4][0]*comp[1] + coeff[5][0]*comp[2] + coeff[7][0]*comp[1]*comp[2],
          coeff[2][0] + coeff[4][0]*comp[0] + coeff[6][0]*comp[2] + coeff[7][0]*comp[0]*comp[2],
          coeff[3][0] + coeff[5][0]*comp[0] + coeff[6][0]*comp[1] + coeff[7][0]*comp[0]*comp[1],
        },
        {
          coeff[1][1] + coeff[4][1]*comp[1] + coeff[5][1]*comp[2] + coeff[7][1]*comp[1]*comp[2],
          coeff[2][1] + coeff[4][1]*comp[0] + coeff[6][1]*comp[2] + coeff[7][1]*comp[0]*comp[2],
          coeff[3][1] + coeff[5][1]*comp[0] + coeff[6][1]*comp[1] + coeff[7][1]*comp[0]*comp[1],
        },
        {
          coeff[1][2] + coeff[4][2]*comp[1] + coeff[5][2]*comp[2] + coeff[7][2]*comp[1]*comp[2],
          coeff[2][2] + coeff[4][2]*comp[0] + coeff[6][2]*comp[2] + coeff[7][2]*comp[0]*comp[2],
          coeff[3][2] + coeff[5][2]*comp[0] + coeff[6][2]*comp[1] + coeff[7][2]*comp[0]*comp[1],
        }
      };
      // for (auto coe : coeff) {
      //   std::cout << "COE: \n" <<coe << "\n";
      // }
      // std::cout << "My COMP \n" << comp << "\n";
      return jac;
    }

    Matrix3f getJacInvComp2Phys(Array3f comp, std::vector<Array3f> coeff) {
      Matrix3f jac = this->getJacComp2Phys(comp, coeff);
      float det = (-jac(0, 0)*jac(1, 1)*jac(2, 2) - jac(0, 1)*jac(1, 2)*jac(2, 0) - jac(0, 2)*jac(1, 0)*jac(2, 1) + 
                    jac(0, 2)*jac(1, 1)*jac(2, 0) + jac(0, 1)*jac(1, 0)*jac(2, 2) + jac(0, 0)*jac(1, 2)*jac(2, 1));

      Matrix3f inv {
        {
          jac(1, 1)*jac(2, 2) - jac(1, 2)*jac(2, 1),
          -jac(0, 1)*jac(2, 2) + jac(0, 2)*jac(2, 1),
          jac(0, 1)*jac(1, 2) - jac(0, 2)*jac(1, 1),
        },
        {
          -jac(1, 0)*jac(2, 2) + jac(1, 2)*jac(2, 0),
          jac(0, 0)*jac(2, 2) - jac(0, 2)*jac(2, 0),
          -jac(0, 0)*jac(1, 2) + jac(0, 2)*jac(1, 0),
        },
        {
          jac(1, 0)*jac(2, 1) - jac(1, 1)*jac(2, 0),
          jac(0, 0)*jac(2, 1) - jac(0, 1)*jac(2, 0),
          jac(0, 0)*jac(1, 1) - jac(0, 1)*jac(1, 0),
        }
      };
      // std::cout << "My Inv \n" << jac <<"\n" <<  inv <<  "  --  " << det << "\n\n";
      return inv / -det;
    }

    //		    6________7  high-vtx
    //		   /|       /|
    //		  / |      / |
    //		4/_______5/  |
    //		|  2|___ |___|3
    //		|  /     |  /
    //		| /      | /
    //		|/_______|/
    //		0        1
    //  low_vtx

    //
    //		 011_________111  high-vtx
    //		   /|       /|
    //		  / |      / |
    //	001/_____101/  |
    //		|010|___ |___|110
    //		|  /     |  /
    //		| /      | /
    //		|/_______|/
    //	000       100
    // low vtx

    // given the coordinate index of the low corner vtx (v000)
    // return the 8 coordinates (in Eigen Array3f) in the order
    //  000, 100, 010, 110, 001, 101, 011, 111
    virtual std::vector<Array3f> getVoxelCoords(int i, int j, int k) {
      std::vector<int> indices{i, j, k};
      std::vector<Array3f> cell(8);
      cell[0] = this->getCoord(indices[0],     indices[1],     indices[2]);
      cell[1] = this->getCoord(indices[0] + 1, indices[1],     indices[2]);
      cell[2] = this->getCoord(indices[0],     indices[1] + 1, indices[2]);
      cell[3] = this->getCoord(indices[0] + 1, indices[1] + 1, indices[2]);
      cell[4] = this->getCoord(indices[0],     indices[1],     indices[2] + 1);
      cell[5] = this->getCoord(indices[0] + 1, indices[1],     indices[2] + 1);
      cell[6] = this->getCoord(indices[0],     indices[1] + 1, indices[2] + 1);
      cell[7] = this->getCoord(indices[0] + 1, indices[1] + 1, indices[2] + 1);
      return cell;
    }

    // std::vector<Array3f> getVoxelCoords(float x, float y, float z) {
    //   std::vector<int> indices = this->getVoxel(x,y,z);
    //   std::vector<Array3f> cell(8);
    //   cell[0] = this->getCoord(indices[0],     indices[1],     indices[2]);
    //   cell[1] = this->getCoord(indices[0] + 1, indices[1],     indices[2]);
    //   cell[2] = this->getCoord(indices[0],     indices[1] + 1, indices[2]);
    //   cell[3] = this->getCoord(indices[0] + 1, indices[1] + 1, indices[2]);
    //   cell[4] = this->getCoord(indices[0],     indices[1],     indices[2] + 1);
    //   cell[5] = this->getCoord(indices[0] + 1, indices[1],     indices[2] + 1);
    //   cell[6] = this->getCoord(indices[0],     indices[1] + 1, indices[2] + 1);
    //   cell[7] = this->getCoord(indices[0] + 1, indices[1] + 1, indices[2] + 1);
    //   return cell;
    // }

    // return corner Cell: corner x,y,z grid point index
    // can be overriden in
    //  - cartesian
    virtual std::vector<int> getVoxel(float x, float y, float z) {
      return std::vector<int>();
      // std::vector<int> indices(3, 0);
      // std::vector<float> location{ x, y, z };
      // // for each dimension
      // for (int i = 0; i < this->dims.size(); i++) {
      //   // search for the index along this dimension
      //   DimPropertyRectlinear *dim = &this->dims[i];
      //   //
      //   if (dim->phys[0] == location[i]) {
      //     indices[i] = 0;
      //     continue;
      //   }
      //   for (int j = 1; j < dim->len; j++) {
      //     if (dim->phys[j-1] > location[i] && dim->phys[j] <= location[i]) {
      //       indices[i] = j-1;
      //       continue;
      //     }
      //   }
      // }
      // return indices;
    }

    // given the coordinate index of the low corner vtx (v000)
    // return the 8 coordinates (in Eigen Array3f) in the order
    //  000, 100, 010, 110, 001, 101, 011, 111 
    virtual Array3f getCoord(int i, int j, int k) {
      Array3f coord = this->coords.getVal(i, j, k);
      return coord;
    }

    // check if a coordiniate is inside a cell cornerd at i,j,k index
    // bool isBoundedCell(float x, float y, float z, int i, int j, int k) {

    // }

    virtual bool isBounded(float x, float y, float z) {
      // Array3f phys{x,y,z};
      // Array3f comp = this->phys2comp_newtwon(phys, )
      bool bounded = (x >= this->dims[0].min && x <= this->dims[0].max) &&
                     (y >= this->dims[1].min && y <= this->dims[1].max) &&
                     (z >= this->dims[2].min && z <= this->dims[2].max);
      return bounded;
    }

    void findBoundBoxPhysical() {
      std::vector<Vector3f> coordsData = this->coords.getData();
      Array3f mins = coordsData[0];
      Array3f maxs = coordsData[0];
      for (const Vector3f &coord : coordsData) {
        for (int i = 0; i < 3; i++) {
          if (coord(i) < mins(i)) {
            mins(i) = coord(i);
          }
          if (coord(i) > maxs(i)) {
            maxs(i) = coord(i);
          }
        }
      }
      this->bbox.col(0) = mins;
      this->bbox.col(1) = maxs;

      for (int i = 0; i < 3; i++) {
        this->dims[i].min = mins(i);
        this->dims[i].max = maxs(i);
      }

    }
};
/*
Question:
1. how is curvelinear grid implemented? can it inherient regular grid

if treat regular cartesian grid as a grid of uniform spacing

2. given a cell? what is cell?
3. neighboring cell: 6 neighbors or 27 neighbors or what neighbors?

4. okay to assume grid min, max, spacing, # of vertex to be integer?
*/

// *************************************** Rewrite everything with vector

class NeuralCurvilinearGrid: public CurvilinearGrid {

  public:
    torch::jit::script::Module network;
    std::vector<std::vector<glm::vec3>> compRays;

    NeuralCurvilinearGrid() {}
    NeuralCurvilinearGrid(const std::string & modulePath) {
      network = torch::jit::load(modulePath);
    }


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

    std::vector<std::vector<glm::vec3>> raysTensorToVectorGLM(torch::Tensor rays) {
      int shape[3] = {(int)rays.sizes()[0], (int)rays.sizes()[1], (int)rays.sizes()[2]};

      std::vector<std::vector<glm::vec3>> raysSTL;
      raysSTL.resize(shape[0]);
      for (int i = 0; i < shape[0]; i++) {
        std::vector<glm::vec3> raySTL;
        raySTL.resize(shape[1]);
        for (int j = 0; j < shape[1]; j++) {
          glm::vec3 coord(
            rays[i][j][0].item<float>(),
            rays[i][j][1].item<float>(),
            rays[i][j][2].item<float>()
          );
          raySTL[j] = coord;
        }
        raysSTL[i] = raySTL;
      }
      return raysSTL;
    }

    // precompute
    std::vector<std::vector<glm::vec3>> precomputeRayCompCoords(std::vector<std::vector<glm::vec3>> physRays) {
      std::vector<float> raysSTL = flattenVectorGLM(physRays);
      torch::Tensor physRaysTensor = torch::tensor(raysSTL);
      physRaysTensor = physRaysTensor.reshape({-1, 3});
      std::cout << physRaysTensor.sizes() << "\n";
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(physRaysTensor);
      torch::Tensor compRaysTensor = this->network.forward(inputs).toTensor();

      int dims[3] = {(int)physRays.size(), (int)physRays[0].size(), 3};
      compRaysTensor = compRaysTensor.reshape({dims[0], dims[1], dims[2]});
      std::cout << compRaysTensor.sizes() << "\n";
      return raysTensorToVectorGLM(compRaysTensor);
    }

    // virtual CellLerp getVoxelLerp(float x, float y, float z) {
    //   // assert(this->isBounded(x,y,z));
    //   // 2. find comp
    //   // newton's method
    //   Array3f phys{ x, y, z };
    //   Array3f comp = this->phys2comp_newtwon(
    //     phys, voxelCoords, this->newton.maxiter, this->newton.atol, this->newton.rtol
    //   );
    //   std::vector<float> weights = { comp(0), comp(1), comp(2) };
    //   std::vector<int> ijk = vtkGetIJK(cornerPID);
    //   // 3. trilinear interpolate in comp grid
    //   return CellLerp{ ijk, weights };
    // }
};

#endif