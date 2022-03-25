#ifndef SVK_CONTRAINER_GRID_BASE_H
#define SVK_CONTRAINER_GRID_BASE_H


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
struct CellLerp {
  // Voxel representation: lower left xyz index, and interpolation weights
  std::vector<int> indices;
  std::vector<float> weights;
};

struct CellSysEqLerp {
  // Voxel representation: lower left xyz index, and interpolation weights
  std::vector<int> indices;
  std::vector<float> weights;
};

class DimPropertyBase {
  public:
    // float EPS = 1e-8;
    float min, max;
    int len;
    // int order;
    // int stride;
    DimPropertyBase() {}
    ~DimPropertyBase(){}

    // virtual int getIndex(float x, float y, float z);
};

// template <typename T>
class GridBase {
  private:

  public:
    // xyzorder: order of axis. Ex: 
    GridBase() {}
    ~GridBase() {}

    virtual int getCellCount() = 0;
    virtual int getVtxCount() = 0;
    virtual int getDimCount() = 0;

    virtual std::vector<float> getDomain(int idim) = 0;
    virtual int getDimLen(int idim) = 0;


    // ************************************************************************
    // core functions

    // return corner Cell: corner x,y,z grid point index
    virtual std::vector<int> getVoxel(float x, float y, float z) = 0;
    virtual CellLerp getVoxelLerp(float x, float y, float z) = 0;

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
    virtual bool isBounded(float x, float y, float z) = 0;
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