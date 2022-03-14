#ifndef GRID_H
#define GRID_H

#include "common.h"
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

class DimProperty {
  public:
    // double EPS = 1e-8;
    double min, max;
    double spacing;
    int len;
    // int order;
    // int stride;
    DimProperty() {}
    DimProperty(double min, double max, double spacing): min(min), max(max),
                                                      spacing(spacing) {
      // make sure grid resolution is integer
      double tmp = (max-min) / spacing;
      // std::cout << min << " and " << max << " and " << spacing << " and " << tmp << "\n";
      // due to numerical issue, int check would work (e.g. 2 / (2. / 255.) != 255)
      // std::cout << int(tmp) << " and " << static_cast<int>(tmp) << " and " << std::round(tmp) << " and " << (tmp - std::round(tmp) == 0) << "\n";
      // std::cout << std::fabs(tmp - std::round(tmp)) << "\n";
      // assert(std::fabs(tmp - std::round(tmp)) < EPS);
      this->len = std::round(tmp) + 1;
    }
};

// template <typename T>
class Grid {
  private:

  public:
    std::string gtype = "Regular Cartesian";
    std::vector<DimProperty> dims;
    // cell count, vertex count
    int ccount, vcount;

    // xyzorder: order of axis. Ex: 
    Grid() {}
    // Constructor 1: specific dim min max spacing
    Grid(double xmin, double xmax, double ymin,
         double ymax, double zmin, double zmax,
         double xspacing=1., double yspacing=1., double zspacing=1.) {
      this->dims.reserve(3);
      this->dims[0] = DimProperty(xmin, xmax, xspacing);
      this->dims[1] = DimProperty(ymin, ymax, yspacing);
      this->dims[2] = DimProperty(zmin, zmax, zspacing);

      vcount = this->dims[0].len * this->dims[1].len * this->dims[2].len;
      ccount = (this->dims[0].len - 1) * (this->dims[1].len - 1) * (this->dims[2].len - 1);
    }
    // Constructor 2: non-negative indexed regular cartesian grid of domain (0, dim-1)
    Grid(int xdim, int ydim, int zdim) {
      this->dims.reserve(3);
      this->dims[0] = DimProperty(0, xdim-1, 1);
      this->dims[1] = DimProperty(0, ydim-1, 1);
      this->dims[2] = DimProperty(0, zdim-1, 1);

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
    double getDimSpacing(int idim) { return this->dims[idim].spacing; }

    // ************************************************************************
    // core functions

    // return floor corner x,y,z index and lerp weights on x,y,z
    CellLerp getVoxelLerp(double x, double y, double z) {
      assert(this->isBounded(x,y,z));

      std::vector<int> indices(3, 0);
      std::vector<double> weights(3, 0);
      std::vector<double> location{ x, y, z };
      double whole;
      for (int i = 0; i < 3; i++) {
        double index = (location[i] - this->dims[i].min) / this->dims[i].spacing;
        indices[i] = int(index);
        weights[i] = index - indices[i];
      }
      CellLerp cl = { indices, weights };
      return cl;
    }


    // return corner Cell: corner x,y,z grid point index
    std::vector<int> getVoxel(double x, double y, double z) {
      std::vector<int> indices(3, 0);
      std::vector<double> location{ x, y, z };
      for (int i = 0; i < 3; i++) {
        double index = (location[i] - this->dims[i].min) / this->dims[i].spacing;
        indices[i] = floor(index);
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