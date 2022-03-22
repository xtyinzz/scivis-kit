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

class DimPropertyCurvilinear: public DimPropertyBase {
  public:
    // double EPS = 1e-8;
    std::vector<double> phys;
    // int order;
    // int stride;
    DimPropertyCurvilinear() {}
    DimPropertyCurvilinear(std::vector<double> coords): phys(coords) {
      this->min = this->phys[0];
      this->max = this->phys.back();
      this->len = this->phys.size();
    }
};

// template <typename T>
class CurvilinearGrid: public GridBase {
  private:

  public:
    std::string gtype = "Curvilinear";
    std::vector<DimPropertyCurvilinear> dims;
    

    // cell count, vertex count
    int ccount, vcount;

    // xyzorder: order of axis. Ex: 
    CurvilinearGrid() {}

    // Constructor 2: non-negative indexed regular cartesian grid of domain (0, dim-1)
    CurvilinearGrid(std::vector<double> *xCoords, std::vector<double> *yCoords, std::vector<double> *zCoords) {
      this->dims.reserve(3);
      this->dims[0] = DimPropertyCurvilinear(*xCoords);
      this->dims[1] = DimPropertyCurvilinear(*yCoords);
      this->dims[2] = DimPropertyCurvilinear(*zCoords);

      vcount = this->dims[0].len * this->dims[1].len * this->dims[2].len;
      ccount = (this->dims[0].len - 1) * (this->dims[1].len - 1) * (this->dims[2].len - 1);
    }

    int getCellCount() { return this->ccount; }
    int getVtxCount() { return this->vcount; }
    int getDimCount() { return this->dims.size(); } // what's return "dimensions of the grid" in hw write-up?

    std::vector<double> getDomain(int idim) {
      return {this->dims[idim].min, this->dims[idim].max};
    }

    int getDimLen(int idim) { return this->dims[idim].len; }
    void setDimCoord(int idim, std::vector<double> *coords) { this->dims[idim].phys = *coords; }

    // ************************************************************************
    // core functions

    // return floor corner x,y,z index and lerp weights on x,y,z
    CellLerp getVoxelLerp(double x, double y, double z) {
      assert(this->isBounded(x,y,z));

      std::vector<int> indices = this->getVoxel(x, y, z);
      std::vector<double> weights(3, 0);
      std::vector<double> location{ x, y, z };
      double whole;
      for (int i = 0; i < 3; i++) {

      }
      CellLerp cl = { indices, weights };
      return cl;

      // 1. find comp
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