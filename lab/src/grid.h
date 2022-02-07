#ifndef GRID_H
#define GRID_H

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
  std::vector<float> weights;
};

// class Cell {
//   public:
//     int xi, yi, zi;
//     float alpha, beta, gamma;

//     Cell() {}
//     Cell(int xi, int yi, int zi): xi(xi), yi(yi), zi(zi) {}

//     void calcLerpWeights(float x, float y, float z) {
//       this->alpha = (x - this->xi) / this->x.spacing;
//       this->beta = (y - this->yi) / this->y.spacing;
//       this->gamma = (z - this->zi) / this->z.spacing;
//     }
// }

class DimProperty {
  public:
    float min, max;
    float spacing;
    int len;
    // int order;
    int stride;
    DimProperty() {}
    DimProperty(float min, float max, float spacing): min(min), max(max),
                                                      spacing(spacing), stride(1) {
      // make sure grid resolution is integer
      float tmp = (max-min) / spacing;
      // due to numerical issue, int check would work (e.g. 2 / (2. / 255.) != 255)
      assert((tmp - int(tmp) == 0));
      len = (int)tmp + 1;
    }
};

// template <typename T>
class Grid {

  public:
    std::string gtype = "Regular Cartesian";
    std::vector<DimProperty> dims;
    // cell count, vertex count
    int ccount, vcount;

    // xyzorder: order of axis. Ex: 
    Grid() {}
    // Constructor 1: specific dim min max spacing
    Grid(float xmin, float xmax, float ymin,
         float ymax, float zmin, float zmax,
         float xspacing=1., float yspacing=1., float zspacing=1.) {
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
      this->dims[0] = DimProperty(0, xdim, 1);
      this->dims[1] = DimProperty(0, ydim, 1);
      this->dims[2] = DimProperty(0, zdim, 1);

      vcount = this->dims[0].len * this->dims[1].len * this->dims[2].len;
      ccount = (this->dims[0].len - 1) * (this->dims[1].len - 1) * (this->dims[2].len - 1);
    }

    int cellCount() { return this->ccount; }
    int vtxCount() { return this->vcount; }
    int dimCount() { return 3; } // what's return "dimensions of the grid" in hw write-up?

    std::vector<float> domain(int idim) {
      return {this->dims[idim].min, this->dims[idim].max};
    }

    int dimLength(int idim) { return this->dims[idim].len; }

    // ************************************************************************
    // core functions

    // return floor corner x,y,z index and lerp weights on x,y,z
    CellLerp getVoxelLerp(float x, float y, float z) {
      assert(this->isBounded(x,y,z));

      std::vector<int> indices(3, 0);
      std::vector<float> weights(3, 0);
      std::vector<float> location{ x, y, z };
      float whole;
      for (int i = 0; i < 3; i++) {
        float index = (location[i] - this->dims[i].min) / this->dims[i].spacing;
        indices[i] = floor(index);
        weights[i] = std::modf(index, &whole);
      }
      CellLerp cl = { indices, weights };
      return cl;
    }


    // return corner Cell: corner x,y,z grid point index
    std::vector<int> getVoxel(float x, float y, float z) {
      std::vector<int> indices(3, 0);
      std::vector<float> location{ x, y, z };
      for (int i = 0; i < 3; i++) {
        float index = (location[i] - this->dims[i].min) / this->dims[i].spacing;
        indices[i] = floor(index);
      }
      return indices;
    }

    // std::vector<float> getLerpWeights(float x, float y, float z) {
    //   std::vector<float> w(3, 0);
    //   std::vector<float> location{ x, y, z };
    //   float whole;
    //   for (int i = 0; i < 3; i++) {
    //     float index = (location[i] - this->dims[i].min) / this->dims[i].spacing;
    //     w[i] = modf(index, &whole);
    //   }
    // }

    // struct Vertex[] getVoxelNeigbor(float x, float y, float z);
    bool isBounded(float x, float y, float z) {
      std::vector<float> location{ x, y, z };
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