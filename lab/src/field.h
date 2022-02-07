#ifndef FIELD_H
#define FIELD_H

#include "grid.h"
#include "solution.h"

template <typename T=float>
class Field {
  public:
    Grid *g;
    Solution<T> *s;

    Field() {}
    // $$$$$$$$$$$$$$$$$ CAN'T INITIALIZE G&S and point to it b/c scope. IF NEEDED, REFACTORATION NEEDED.
    // Field(float xmin, float xmax, float ymin,
    //       float ymax, float zmin, float zmax,
    //       float xspacing=1., float yspacing=1., float zspacing=1.) {
    //   Grid gtmp(xmin, xmax, ymin, ymax, zmin, zmax, xspacing, yspacing, zspacing);
    //   Solution<T> stmp(gtmp.dimLength(0), gtmp.dimLength(1), gtmp.dimLength(2));
    //   std::cout << gtmp.dimLength(0) << " " << gtmp.dimLength(1) << " " << gtmp.dimLength(2) << "\n";

    //   this->g = &gtmp;
    //   this->s = &stmp;
    // }
    // Field(int xdim, int ydim, int zdim) {
    //   Grid gtmp(xdim, ydim, zdim);
    //   Solution<T> stmp(gtmp.dimLength(0), gtmp.dimLength(1), gtmp.dimLength(2));

    //   this->g = &gtmp;
    //   this->s = &stmp;
    // }
    
    Field(Grid *g): g(g) {}
    Field(Solution<T> *s): g(s) {}
    Field(Grid *g, Solution<T> *s): g(g), s(s) {} 

    Grid *grid() { return this->g; }
    Solution<T> *solution() { return this->s; }
    void setGrid(Grid *g) { this->g = g; }
    void setSolution(Solution<T> *s) { this->s = s; }

    // get extents
    std::vector<float> dimExtent(int idim) {
      return this->g->domain(idim);
    }
    int dimLength(int idim) {
      return this->g->dimLength(idim);
    }
    // std::vector<T> range() {
    //   return s->range();
    // }

    // check if location is withtin bound
    bool isBounded(float x, float y, float z) { return this->g->isBounded(x, y, z); }

    void setVal(int xi, int yi, int zi, T val) {
      this->s->setVal(xi, yi, zi, val);
    }

    // get value at location
    T val(float x, float y, float z) {
      CellLerp cl = this->g->getVoxelLerp(x, y, z);
      return this->s->val_trilerp(cl.indices[0], cl.indices[1], cl.indices[2],
                                  cl.weights[0], cl.weights[1], cl.weights[2]);
    }
    // get gradient at location
    std::vector<T> grad(float x, float y, float z) {
      CellLerp cl = this->g->getVoxelLerp(x, y, z);
      return this->s->grad_trilerp(cl.indices[0], cl.indices[1], cl.indices[2],
                                  cl.weights[0], cl.weights[1], cl.weights[2]);
    }
    
};

#endif