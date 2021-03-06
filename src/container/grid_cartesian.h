#ifndef SVK_CONTRAINER_GRID_CARTESIAN_H
#define SVK_CONTRAINER_GRID_CARTESIAN_H


#include "grid_base.h"
#include <assert.h>
#include <iostream>
#include <string>
#include <cmath>
#include <assert.h>
#include <vector>



class DimPropertyCartesian: public DimPropertyBase {
  public:
    // float EPS = 1e-8;
    float spacing;
    // int order;
    // int stride;
    DimPropertyCartesian() {}
    DimPropertyCartesian(float min, float max, float spacing) {
      this->min = min;
      this->max = max;
      this->spacing = spacing;
      // make sure grid resolution is integer
      float tmp = (max-min) / spacing;
      // std::cout << min << " and " << max << " and " << spacing << " and " << tmp << "\n";
      // due to numerical issue, int check would work (e.g. 2 / (2. / 255.) != 255)
      // std::cout << int(tmp) << " and " << static_cast<int>(tmp) << " and " << std::round(tmp) << " and " << (tmp - std::round(tmp) == 0) << "\n";
      // std::cout << std::fabs(tmp - std::round(tmp)) << "\n";
      // assert(std::fabs(tmp - std::round(tmp)) < EPS);
      this->len = std::round(tmp) + 1;
    }
};

// template <typename T>
class CartesianGrid: public GridBase {
  private:

  public:
    std::string gtype = "Regular Cartesian";
    std::vector<DimPropertyCartesian> dims;
    // cell count, vertex count
    int ccount = 0;
    int vcount = 0;

    // xyzorder: order of axis. Ex: 
    CartesianGrid() {}

    CartesianGrid(float xmin, float xmax, int xlen, float ymin,
         float ymax, int ylen, float zmin, float zmax, int zlen) {
      float xspacing = (xmax - xmin) / (xlen-1);
      float yspacing = (ymax - ymin) / (ylen-1);
      float zspacing = (zmax - zmin) / (zlen-1);
      this->dims.resize(3);
      this->dims[0] = DimPropertyCartesian(xmin, xmax, xspacing);
      this->dims[1] = DimPropertyCartesian(ymin, ymax, yspacing);
      this->dims[2] = DimPropertyCartesian(zmin, zmax, zspacing);

      vcount = this->dims[0].len * this->dims[1].len * this->dims[2].len;
      ccount = (this->dims[0].len - 1) * (this->dims[1].len - 1) * (this->dims[2].len - 1);
    }

    // Constructor 1: specific dim min max spacing
    CartesianGrid(float xmin, float xmax, float ymin,
         float ymax, float zmin, float zmax,
         float xspacing=1., float yspacing=1., float zspacing=1.) {
      this->dims.resize(3);
      this->dims[0] = DimPropertyCartesian(xmin, xmax, xspacing);
      this->dims[1] = DimPropertyCartesian(ymin, ymax, yspacing);
      this->dims[2] = DimPropertyCartesian(zmin, zmax, zspacing);

      vcount = this->dims[0].len * this->dims[1].len * this->dims[2].len;
      ccount = (this->dims[0].len - 1) * (this->dims[1].len - 1) * (this->dims[2].len - 1);
    }
    // Constructor 2: non-negative indexed regular cartesian grid of domain (0, dim-1)
    CartesianGrid(int xdim, int ydim, int zdim) {
      this->dims.resize(3);
      this->dims[0] = DimPropertyCartesian(0, xdim-1, 1);
      this->dims[1] = DimPropertyCartesian(0, ydim-1, 1);
      this->dims[2] = DimPropertyCartesian(0, zdim-1, 1);

      vcount = this->dims[0].len * this->dims[1].len * this->dims[2].len;
      ccount = (this->dims[0].len - 1) * (this->dims[1].len - 1) * (this->dims[2].len - 1);
    }

    int getCellCount() { return this->ccount; }
    int getVtxCount() { return this->vcount; }
    int getDimCount() { return this->dims.size(); } // what's return "dimensions of the grid" in hw write-up?

    std::vector<float> getDomain(int idim) {
      return {this->dims[idim].min, this->dims[idim].max};
    }

    int getDimLen(int idim) { return this->dims[idim].len; }
    float getDimSpacing(int idim) { return this->dims[idim].spacing; }

    void swapAxes(int idim, int jdim, int kdim) {
      
    }
    // ************************************************************************
    // core functions

    // return floor corner x,y,z index and lerp weights on x,y,z
    CellLerp getVoxelLerp(float x, float y, float z) {
      assert(this->isBounded(x,y,z));

      std::vector<int> indices(3, 0);
      std::vector<float> weights(3, 0);
      // std::vector<float> tmplocation{ x,y,z };
      std::vector<float> location{ x,y,z };
      // location[0] = tmplocation[this->xyzorder[0]];
      // location[1] = tmplocation[this->xyzorder[1]];
      // location[2] = tmplocation[this->xyzorder[2]];
      float whole;
      for (int i = 0; i < 3; i++) {
        float index = (location[i] - this->dims[i].min) / this->dims[i].spacing;
        indices[i] = int(index);
        weights[i] = index - indices[i];
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

    void rescale(float newmin, float newmax) {
      this->dims[0].min = newmin;
      this->dims[0].max = newmax;
      this->dims[0].spacing = (newmax - newmin) / (this->dims[0].len - 1);
      this->dims[1].min = newmin;
      this->dims[1].max = newmax;
      this->dims[1].spacing = (newmax - newmin) / (this->dims[1].len - 1);
      this->dims[2].min = newmin;
      this->dims[2].max = newmax;
      this->dims[2].spacing = (newmax - newmin) / (this->dims[2].len - 1);
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