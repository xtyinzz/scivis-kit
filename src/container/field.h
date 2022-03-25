#ifndef SVK_CONTAINER_FIELD_H
#define SVK_CONTAINER_FIELD_H

// #include "grid.h"
#include "grid_base.h"
#include "grid_cartesian.h"
// #include "grid_base.h"
// #include "grid_base.h"
#include "solution.h"


#include <glm/glm.hpp>
// #include <Eigen/Dense>

#include <omp.h>

template <typename T=float>
class Field {
  public:
    // GridBase *g = NULL;
    // Solution<T> *s = NULL;
    GridBase *g = nullptr;
    Solution<T>*s = nullptr;
    Solution<glm::vec3> *grad = nullptr;

    bool hasGrad = false;

    Field() {}
    // $$$$$$$$$$$$$$$$$ CAN'T INITIALIZE G&S and point to it b/c scope. IF NEEDED, REFACTORATION NEEDED.
    // Field(float xmin, float xmax, float ymin,
    //       float ymax, float zmin, float zmax,
    //       float xspacing=1., float yspacing=1., float zspacing=1.) {
    //   GridBase gtmp(xmin, xmax, ymin, ymax, zmin, zmax, xspacing, yspacing, zspacing);
    //   Solution<T> stmp(gtmp.dimLength(0), gtmp.dimLength(1), gtmp.dimLength(2));
    //   std::cout << gtmp.dimLength(0) << " " << gtmp.dimLength(1) << " " << gtmp.dimLength(2) << "\n";

    //   this->g = &gtmp;
    //   this->s = &stmp;
    // }
    // Field(int xdim, int ydim, int zdim) {
    //   GridBase gtmp(xdim, ydim, zdim);
    //   Solution<T> stmp(gtmp.dimLength(0), gtmp.dimLength(1), gtmp.dimLength(2));

    //   this->g = &gtmp;
    //   this->s = &stmp;
    // }
    
    Field(GridBase *g): g(g) {}
    Field(Solution<T> *s): g(s) {}
    Field(GridBase *g, Solution<T> *s): g(g), s(s) {} 
    ~Field() { 
      delete this->grad;
     }

    GridBase *grid() { return this->g; }
    Solution<T> *solution() { return this->s; }
    void setGrid(GridBase *g) { this->g = g; }
    void setSolution(Solution<T> *s) { this->s = s; }
    void setGradSolution(Solution<glm::vec3> *grad) { this->grad = grad; }


    // get extents
    std::vector<float> getDimExtent(int idim) {
      return this->g->getDomain(idim);
    }
    int getDimLen(int idim) {
      return this->g->getDimLen(idim);
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
    // CAN BE OVERRIDE AS ANALYTICAL
    virtual T getVal(float x, float y, float z) {
      CellLerp cl = this->g->getVoxelLerp(x, y, z);
      // std::cout << "\n\n"<< cl.indices[0] <<"-"<<cl.indices[1]<<"-"<<cl.indices[2] <<"\n" << x << "-" << y << "-" << z << "\n";
      // std::cout << cl.weights[0] <<"-"<<cl.weights[1]<<"-"<<cl.weights[2] << "\n";
      return this->s->getValLerp(cl.indices[0], cl.indices[1], cl.indices[2],
                                  cl.weights[0], cl.weights[1], cl.weights[2]);
    }
    
    // get gradient at location
    // CAN BE OVERRIDE AS ANALYTICAL
    virtual glm::vec3 getGrad(float x, float y, float z) {
      CellLerp cl = this->g->getVoxelLerp(x, y, z);
      if (this->grad != nullptr) {
        return this->grad->getValLerp(cl.indices[0], cl.indices[1], cl.indices[2],
                                                cl.weights[0], cl.weights[1], cl.weights[2]);
      }
      return this->s->getGradLerp(cl.indices[0], cl.indices[1], cl.indices[2],
                                  cl.weights[0], cl.weights[1], cl.weights[2]);
    }

    glm::vec3 getGradFromSolution(float x, float y, float z) {
      CellLerp cl = this->g->getVoxelLerp(x, y, z);
      return this->grad->getValLerp(cl.indices[0], cl.indices[1], cl.indices[2],
                                  cl.weights[0], cl.weights[1], cl.weights[2]);
    }

    T getValByIndex(int xi, int yi, int zi) {
      return this->s->getVal(xi, yi, zi);
    }
    T getValByIndex(int xi, int yi) {
      return this->s->getVal(xi, yi);
    }

    void computeGradSolution3D() {
      this->grad = new Solution<glm::vec3>(
        this->getDimLen(0), this->getDimLen(1), this->getDimLen(2)
      );
      // std::cout << this->grad->getDimLen(0) << " " << this->grad->getDimLen(1) << " " << this->grad->getDimLen(2) << "\n";
      #pragma omp parallel for collapse(3)
      for (int i = 0; i < this->getDimLen(0); i++) {
        for (int j = 0; j < this->getDimLen(1); j++) {
          for (int k = 0; k < this->getDimLen(2); k++) {
            glm::vec3 gradient = this->s->getGrad(i, j, k);
            this->grad->setVal(i, j, k, gradient);
            // printVec(gradient);
          }
        }
      }
      this->hasGrad = true;
    }
    
};



template <typename T=float>
class HeartEquation: public Field<T> {
  public:
    float a, b;

  HeartEquation(float a, float b): a(a), b(b) {}

  T getVal (float x, float y, float z) override {
    float t1 = x*x + ((1.f+this->b)*y)*((1+this->b)*y) + z*z - 1;
    float t2 = -x*x*z*z*z - this->a*y*y*z*z*z;
    return  t1*t1*t1 + t2;
  }

  glm::vec3 getGrad(float x, float y, float z) override {
    glm::vec3 g;
    g[0] = 6.f*x*(x*x + y*y*(1+b)*(1+b)+z*z-1)*(x*x + y*y*(1+b)*(1+b)+z*z-1)-2*x*z*z*z;
    g[1] = 6.f*(1+this->b)*(1+this->b)*y*
           (x*x+y*y*(1+this->b)*(1+this->b)+z*z+1)*(x*x+y*y*(1+this->b)*(1+this->b)+z*z+1) -
           2*this->a*y*z*z*z;
    g[2] = 6.f*z*(x*x+y*y*(1+this->b)*(1+this->b)+z*z+1)*(x*x+y*y*(1+this->b)*(1+this->b)+z*z+1) - 
           3*x*x*z*z - 3*this->a*y*y*z*z;
    return g;
  }

};

#endif