#ifndef SVK_CONTRAINER_GRID_RECTLINEAR_H
#define SVK_CONTRAINER_GRID_RECTLINEAR_H

#include "grid_base.h"
#include <assert.h>
#include <iostream>
#include <string>
#include <cmath>
#include <assert.h>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

class DimPropertyRectlinear: public DimPropertyBase {
  public:
    std::vector<float> phys;

    DimPropertyRectlinear() {}
    DimPropertyRectlinear(std::vector<float> coords): phys(coords) {
      this->updateProperty();
    }
    inline float getPhys(int index) {
      return this->phys[index];
    }
    void setPhys(std::vector<float> coords) {
      this->phys = coords;
      this->updateProperty();
    }
    virtual void updateProperty() {
      this->len = this->phys.size();
      if (this->len > 0) {
        this->min = this->phys[0];
        this->max = this->phys.back();
      }
    }
};


class RectlinearGrid: public GridBase {
  private:

  public:
    std::string gtype;
    std::vector<DimPropertyRectlinear> dims;
    int ccount = 0;
    int vcount = 0;

    RectlinearGrid() {}
    RectlinearGrid(int numDims) {
      this->dims.resize(numDims);
    }
    RectlinearGrid(std::vector<float> *xCoords, std::vector<float> *yCoords, std::vector<float> *zCoords) {
      this->dims.resize(3);
      this->dims[0] = DimPropertyRectlinear(*xCoords);
      this->dims[1] = DimPropertyRectlinear(*yCoords);
      this->dims[2] = DimPropertyRectlinear(*zCoords);

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

    std::vector<float> getDomain(int idim) {
      return { this->dims[idim].min, this->dims[idim].max };
    }

    int getDimLen(int idim) { return this->dims[idim].len; }

    void setPhys(int idim, std::vector<float> *coords) {
      this->dims[idim].setPhys(*coords);
      this->updateCounts();
    }

    void setPhys(std::vector<float> *xcoords, std::vector<float> *ycoords) {
      this->dims[0].setPhys(*xcoords);
      this->dims[1].setPhys(*ycoords);
      this->updateCounts();
    }
    void setPhys(std::vector<float> *xcoords, std::vector<float> *ycoords, std::vector<float> *zcoords) {
      this->dims[0].setPhys(*xcoords);
      this->dims[1].setPhys(*ycoords);
      this->dims[2].setPhys(*zcoords);
      this->updateCounts();
    }

    // ************************************************************************
    // core functions
    // return floor corner x,y,z index and lerp weights on x,y,z
    // can be overriden in subclass
    //  - curvilinear
    //  - cartesian
    CellLerp getVoxelLerp(float x, float y, float z) {
      assert(this->isBounded(x,y,z));

      std::vector<int> indices = this->getVoxel(x, y, z);
      std::vector<float> weights(this->dims.size(), 0);
      std::vector<float> location{ x, y, z };
      for (int i = 0; i < this->dims.size(); i++) {
        int dimcoord0 = this->dims[i].phys[indices[i]];
        int dimcoord1 = this->dims[i].phys[indices[i+1]];
        weights[i] = (location[i] - dimcoord0) / (dimcoord1 - dimcoord0);
      }
      CellLerp cl = { indices, weights };
      return cl;
    }


    // return corner Cell: corner x,y,z grid point index
    // can be overriden in
    //  - cartesian
    std::vector<int> getVoxel(float x, float y, float z) {
      std::vector<int> indices(3, 0);
      std::vector<float> location{ x, y, z };
      // for each dimension
      for (int i = 0; i < this->dims.size(); i++) {
        // search for the index along this dimension
        DimPropertyRectlinear *dim = &this->dims[i];
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
    std::vector<Array3f> getVoxelCoords(int i, int j, int k) {
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

    // given the coordinate index of the low corner vtx (v000)
    // return the 8 coordinates (in Eigen Array3f) in the order
    //  000, 100, 010, 110, 001, 101, 011, 111 
    Array3f getCoord(int i, int j, int k) {
      Array3f coord {
        this->dims[0].phys[i],
        this->dims[1].phys[j],
        this->dims[2].phys[k], 
      };
      return coord;
    }
    
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