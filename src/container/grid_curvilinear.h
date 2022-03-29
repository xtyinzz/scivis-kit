#ifndef SVK_CONTRAINER_GRID_CURVILINEAR_H
#define SVK_CONTRAINER_GRID_CURVILINEAR_H

// #include "common.h"
#include "grid_base.h"
#include "grid_rectlinear.h"
#include "solution.h"
#include "common/numerical.h"
#include <assert.h>
#include <iostream>
#include <string>
#include <cmath>
#include <assert.h>
#include <vector>
#include <Eigen/Dense>

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


    // TODO
    void updateCounts() {
      // vcount = this->dims[0].len * this->dims[1].len * this->dims[2].len;
      // ccount = (this->dims[0].len - 1) * (this->dims[1].len - 1) * (this->dims[2].len - 1);
    }

    int getCellCount() { return this->ccount; }
    int getVtxCount() { return this->vcount; }
    int getDimCount() { return this->dims.size(); } // what's return "dimensions of the grid" in hw write-up?

    std::vector<float> getDomain(int idim) {
      return { this->dims[idim].min, this->dims[idim].max };
    }

    int getDimLen(int idim) { return this->dims[idim].len; }

    // ************************************************************************
    // core functions

    void setCompGrid(RectlinearGrid *compGrid) {
      this->compGrid = compGrid;
    }

    // phys2comp
    // Given phys coord, output cell indices and comp space lerp weights by Newton's Method
    // return floor corner x,y,z index and lerp weights on x,y,z
    virtual CellLerp getVoxelLerp(float x, float y, float z) {
      assert(this->isBounded(x,y,z));

      // 1. locate the cell
      std::vector<int> indices = this->getVoxel(x, y, z);
      std::vector<int> voxelIndices = this->getVoxel(x, y, z);
      std::vector<Array3f> compVoxelCoords = this->compGrid->getVoxelCoords(
        voxelIndices[0], voxelIndices[1], voxelIndices[2]
      );
      std::vector<Array3f> voxelCoords = this->getVoxelCoords(
        voxelIndices[0], voxelIndices[1], voxelIndices[2]
      );
      Array3f phys{ x, y, z };

      // 2. find comp
      // newton's method
      Array3f comp = this->phys2comp_newtwon(
        phys, compVoxelCoords[0], compVoxelCoords[7], voxelCoords,
        this->newton.maxiter, this->newton.atol, this->newton.rtol
      );
      // 3. trilinear interpolate in comp grid
      return this->compGrid->getVoxelLerp(comp[0], comp[1], comp[2]);
    }


    Array3f phys2comp_newtwon(const Array3f &phys, const Array3f &lowVtx, const Array3f &highVtx, std::vector<Array3f> physVoxelCoord,
    int maxiter=50, float atol=1.48e-8, float rtol=0.0) {
      Array3f init_comp = (lowVtx + highVtx) / 2.;
      Array3f comp = init_comp;
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
        // std::cout << "phys_est: " << phys_est << "comp: " << new_comp << "error: " << abs(diff_funcval.array()) << "\n";
        
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
      return inv / det;
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
      return false;
      // Array3f phys{x,y,z};
      // Array3f comp = this->phys2comp_newtwon(phys, )


      // std::vector<float> location{ x, y, z };
      // bool bounded = (x >= this->dims[0].min && x <= this->dims[0].max) &&
      //                (y >= this->dims[1].min && y <= this->dims[1].max) &&
      //                (z >= this->dims[2].min && z <= this->dims[2].max);
      // return bounded;
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