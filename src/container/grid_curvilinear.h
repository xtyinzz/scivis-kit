#ifndef GRID_CURVILINEAR_H
#define GRID_CURVILINEAR_H

#include "common.h"
#include "grid_base.h"
#include "grid_rectlinear.h"
#include <assert.h>
#include <iostream>
#include <string>
#include <cmath>
#include <assert.h>
#include <vector>

// struct Cell {
//   // Voxel representation: lower left xyz index;
//   int xi, yi, zi;
// };

struct CellLerp {
  // Voxel representation: lower left xyz index, and interpolation weights
  std::vector<int> indices;
  std::vector<double> weights;
};

// class Cell {
//   public:
//     int xi, yi, zi;
//     double alpha, beta, gamma;

//     Cell() {}
//     Cell(int xi, int yi, int zi): xi(xi), yi(yi), zi(zi) {}

//     void calcLerpWeights(double x, double y, double z) {
//       this->alpha = (x - this->xi) / this->x.spacing;
//       this->beta = (y - this->yi) / this->y.spacing;
//       this->gamma = (z - this->zi) / this->z.spacing;
//     }
// }

class DimPropertyCurvilinear: public DimPropertyRectlinear {
  public:
    // double EPS = 1e-8;
    // std::vector<double> comp;
    // int order;
    // int stride;
    DimPropertyCurvilinear() {}
    DimPropertyCurvilinear(std::vector<double> coords): DimPropertyRectlinear(coords) { }
};

// template <typename T>
class CurvilinearGrid: public GridBase {
  private:

  public:
    std::string gtype = "Curvilinear";
    std::vector<DimPropertyCurvilinear> dims;
    RectlinearGrid *compGrid;
    

    // cell count, vertex count
    int ccount = 0;
    int vcount = 0;

    // xyzorder: order of axis. Ex: 
    CurvilinearGrid() {}
    CurvilinearGrid(int numDims) {
      this->dims.resize(numDims);
    }
    // Constructor 2: non-negative indexed regular cartesian grid of domain (0, dim-1)
    CurvilinearGrid(std::vector<double> *xCoords, std::vector<double> *yCoords, std::vector<double> *zCoords) {
      this->dims.resize(3);
      this->dims[0] = DimPropertyCurvilinear(*xCoords);
      this->dims[1] = DimPropertyCurvilinear(*yCoords);
      this->dims[2] = DimPropertyCurvilinear(*zCoords);

      vcount = this->dims[0].len * this->dims[1].len * this->dims[2].len;
      ccount = (this->dims[0].len - 1) * (this->dims[1].len - 1) * (this->dims[2].len - 1);
    }

        // TODO
    void updateCounts() {
      // vcount = this->dims[0].len * this->dims[1].len * this->dims[2].len;
      // ccount = (this->dims[0].len - 1) * (this->dims[1].len - 1) * (this->dims[2].len - 1);
    }

    int getCellCount() { return this->ccount; }
    int getVtxCount() { return this->vcount; }
    int getDimCount() { return this->dims.size(); } // what's return "dimensions of the grid" in hw write-up?

    std::vector<double> getDomain(int idim) {
      return {this->dims[idim].min, this->dims[idim].max};
    }

    int getDimLen(int idim) { return this->dims[idim].len; }
    void setPhys(int idim, std::vector<double> *coords) {
      this->dims[idim].setPhys(*coords);
      this->updateCounts();
    }
    void setPhys(std::vector<double> *xcoords, std::vector<double> *ycoords) {
      this->dims[0].setPhys(*xcoords);
      this->dims[1].setPhys(*ycoords);
      this->updateCounts();
    }
    void setPhys(std::vector<double> *xcoords, std::vector<double> *ycoords, std::vector<double> *zcoords) {
      this->dims[0].setPhys(*xcoords);
      this->dims[1].setPhys(*ycoords);
      this->dims[2].setPhys(*zcoords);
      this->updateCounts();
    }

    void setCompGrid(RectlinearGrid *compGrid) {
      this->compGrid = compGrid;
    }


    // ************************************************************************
    // core functions

    // return floor corner x,y,z index and lerp weights on x,y,z
    // Given phys coord, output cell indices and comp space lerp weights by Newton's Method
    CellLerp getVoxelLerp(double x, double y, double z, double tol=1.48e-8, int maxiter=50, double rtol=0.0) {
      assert(this->isBounded(x,y,z));

      std::vector<int> indices = this->getVoxel(x, y, z);
      std::vector<Array3d> compVoxelCoord = this->compGrid->getVoxelCoords(x, y, z);
      Array3d lowVtx = compVoxelCoord[0];
      Array3d highVtx = compVoxelCoord[1];
      Array3d phys{ x, y, z };

      // newton's method
      Array3d init_comp = (lowVtx + highVtx) / 2.;
      Array3d comp = init_comp;
      Array3d goal_diff{0., 0., 0.};
      for(int i = 0; i < maxiter; i++) {
        std::vector<Array3d> coeff = trilerpSysEqCoeff(comp, lowVtx, highVtx, compVoxelCoord);
        Array3d phys_est = trilerpSysEq(comp[0], comp[1], comp[2], coeff);
        Matrix3Xd diff_funcval = phys_est - phys;

        Matrix3d jac_inv = this->getJacInvComp2Phys(comp, coeff);
        Array3d new_comp = comp - (jac_inv * diff_funcval).array();
      }

      

      std::vector<double> weights(3, 0);
      std::vector<double> location{ x, y, z };
      double whole;
      for (int i = 0; i < 3; i++) {

      }
      CellLerp cl = { indices, weights };
      return cl;
      // 1. find comp
    }

    Matrix3d getJacComp2Phys(Array3d comp, std::vector<Array3d> coeff) {
      return Matrix3d {
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
    }

    Matrix3d getJacInvComp2Phys(Array3d comp, std::vector<Array3d> coeff) {
      Matrix3d jac = this->getJacComp2Phys(comp, coeff);
      double det = (-jac[0,0]*jac[1,1]*jac[2,2] - jac[0,1]*jac[1,2]*jac[2,0] - jac[0,2]*jac[1,0]*jac[2,1] + 
                    jac[0,2]*jac[1,1]*jac[2,0] + jac[0,1]*jac[1,0]*jac[2,2] + jac[0,0]*jac[1,2]*jac[2,1]);

      Matrix3d inv {
        {
          jac[1,1]*jac[2,2] - jac[1,2]*jac[2,1],
          -jac[0,1]*jac[2,2] + jac[0,2]*jac[2,1],
          jac[0,1]*jac[1,2] - jac[0,2]*jac[1,1],
        },
        {
          -jac[1,0]*jac[2,2] + jac[1,2]*jac[2,0],
          jac[0,0]*jac[2,2] - jac[0,2]*jac[2,0],
          -jac[0,0]*jac[1,2] + jac[0,2]*jac[1,0],
        },
        {
          jac[1,0]*jac[2,1] - jac[1,1]*jac[2,0],
          jac[0,0]*jac[2,1] - jac[0,1]*jac[2,0],
          jac[0,0]*jac[1,1] - jac[0,1]*jac[1,0],
        }
      };
      return inv / det;
    }

    // return corner Cell: corner x,y,z grid point index
    std::vector<int> getVoxel(double x, double y, double z) {
      std::vector<int> indices(3, 0);
      std::vector<double> location{ x, y, z };
      // for each dimension
      for (int i = 0; i < this->dims.size(); i++) {
        // search for the index along this dimension
        DimPropertyCurvilinear *dim = &this->dims[i];
        //
        if (dim->phys[0] == location[i]) {
          indices[i] = 0;
          continue;
        }
        for (int j = 1; j < dim->len; j++) {
          if (dim->phys[j-1] > location[i] && dim->phys[j] <= location[i]) {
            indices[i] = j-1;
            continue;
          }
        }
      }
      return indices;
    }

    // std::vector<double> getLerpWeights(double x, double y, double z) {
    //   std::vector<double> w(3, 0);
    //   std::vector<double> location{ x, y, z };
    //   double whole;
    //   for (int i = 0; i < 3; i++) {
    //     double index = (location[i] - this->dims[i].min) / this->dims[i].spacing;
    //     w[i] = modf(index, &whole);
    //   }
    // }

    // struct Vertex[] getVoxelNeigbor(double x, double y, double z);
    bool isBounded(double x, double y, double z) {
      std::vector<double> location{ x, y, z };
      bool bounded = (x >= this->dims[0].min && x <= this->dims[0].max) &&
                     (y >= this->dims[1].min && y <= this->dims[1].max) &&
                     (z >= this->dims[2].min && z <= this->dims[2].max);
      return bounded;
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

#endif