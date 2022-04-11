#ifndef SVK_CONTAINER_FIELD_H
#define SVK_CONTAINER_FIELD_H

#include "grid_base.h"
#include "grid_cartesian.h"
#include "solution.h"
#include "common/numerical.h"

#include <omp.h>

#include <glm/glm.hpp>
#include <Eigen/Dense>

using namespace Eigen;

template <typename T=float>
class FieldBase {
  public:
    // GridBase *g = NULL;
    // Solution<T> *s = NULL;
    GridBase *g = nullptr;
    Solution<T>*s = nullptr;

    FieldBase() {}
    FieldBase(GridBase *g): g(g) {}
    FieldBase(Solution<T> *s): g(s) {}
    FieldBase(GridBase *g, Solution<T> *s): g(g), s(s) {} 


    GridBase *grid() { return this->g; }
    Solution<T> *solution() { return this->s; }
    void setGrid(GridBase *g) { this->g = g; }
    void setSolution(Solution<T> *s) { this->s = s; }

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
      if (! this->isBounded(x, y, z)) {
        return T(0);
      }
      CellLerp cl = this->g->getVoxelLerp(x, y, z);
      // std::cout << "\n\n"<< cl.indices[0] <<"-"<<cl.indices[1]<<"-"<<cl.indices[2] <<"\n" << x << "-" << y << "-" << z << "\n";
      // std::cout << cl.weights[0] <<"-"<<cl.weights[1]<<"-"<<cl.weights[2] << "\n";

      // out of bound
      // if (cl.indices[0] == -1) return NULL;
      return this->s->getValLerp(cl.indices[0], cl.indices[1], cl.indices[2],
                                  cl.weights[0], cl.weights[1], cl.weights[2]);
    }
  
    T getValByIndex(int xi, int yi, int zi) {
      return this->s->getVal(xi, yi, zi);
    }
    T getValByIndex(int xi, int yi) {
      return this->s->getVal(xi, yi);
    }
};

template <typename T=float>
class ScalarField: public FieldBase<T> {
  public:
    Solution<glm::vec3> *grad = nullptr;
    bool hasGrad = false;

    ScalarField() {}
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
    
    ScalarField(GridBase *g): FieldBase<T>(g) {}
    ScalarField(Solution<T> *s): FieldBase<T>(s) {}
    ScalarField(GridBase *g, Solution<T> *s): FieldBase<T>(g, s) {} 
    ~ScalarField() { 
      delete this->grad;
     }
    void setGradSolution(Solution<glm::vec3> *grad) { this->grad = grad; }


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

    std::vector<T> getMinMax() {
      std::vector<T> scalarSol = this->s->getData();
      return std::vector<T>{
        *min_element(scalarSol.begin(), scalarSol.end()),
        *max_element(scalarSol.begin(), scalarSol.end())
      };
    }
};


template <typename T=Array3f>
class VectorField : public FieldBase<T> {
  public:
    Solution<Matrix3f> *jacSol = nullptr;
    Solution<Vector3f> *vortSol = nullptr;
    Solution<float> *vortMagSol = nullptr;

    VectorField() {};
    VectorField(GridBase *g, Solution<T> *s): FieldBase<T>(g, s) {}
    ~VectorField() { 
      delete this->jacSol;
      delete this->vortSol;
      delete this->vortMagSol;
     }

  std::vector<T> particleTracingRK1(T seed, float stepsize, int maxstep) {
    std::vector<T> traces;
    traces.push_back(seed);
    for (int i = 0; i < maxstep; i++) {
      seed = seed + stepsize*(this->getVal(seed[0], seed[1], seed[2]).array());
      // exit if out of bound
      if (!this->isBounded(seed[0], seed[1], seed[2]))
        break;
      traces.push_back(seed);
    }
    return traces;
  }

  void computeJac3D() {
    this->jacSol = new Solution<Matrix3f>(
      this->getDimLen(0), this->getDimLen(1), this->getDimLen(2)
    );
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < this->getDimLen(0); i++) {
      for (int j = 0; j < this->getDimLen(1); j++) {
        for (int k = 0; k < this->getDimLen(2); k++) {
          Matrix3f jacobian = this->s->getJac(i, j, k);
          this->jacSol->setVal(i, j, k, jacobian);
        }
      }
    }
  }

  void computeVort3D() {
    this->vortSol = new Solution<Vector3f>(
      this->getDimLen(0), this->getDimLen(1), this->getDimLen(2)
    );
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < this->getDimLen(0); i++) {
      for (int j = 0; j < this->getDimLen(1); j++) {
        for (int k = 0; k < this->getDimLen(2); k++) {
          Matrix3f jacobian = this->jacSol->getVal(i, j, k);
          Vector3f vorticity = {
            jacobian(2, 1) - jacobian(1, 2),
            jacobian(0, 2) - jacobian(2, 0),
            jacobian(1, 0) - jacobian(0, 1)
          };
          this->vortSol->setVal(i, j, k, vorticity);
        }
      }
    }
  }

  void computeVortMag() {
    this->vortMagSol = new Solution<float>(
      this->getDimLen(0), this->getDimLen(1), this->getDimLen(2)
    );
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < this->getDimLen(0); i++) {
      for (int j = 0; j < this->getDimLen(1); j++) {
        for (int k = 0; k < this->getDimLen(2); k++) {
          Vector3f vort = this->vortSol->getVal(i, j, k);
          this->vortMagSol->setVal(i, j, k, vort.norm());
        }
      }
    }
  }

  // get gradient at location
  // CAN BE OVERRIDE AS ANALYTICAL
  Matrix3f getJac(float x, float y, float z) {
    CellLerp cl = this->g->getVoxelLerp(x, y, z);
    if (this->jacSol != nullptr) {
      return this->jacSol->getValLerp(cl.indices[0], cl.indices[1], cl.indices[2],
                                              cl.weights[0], cl.weights[1], cl.weights[2]);
    }
    return this->s->getJacLerp(cl.indices[0], cl.indices[1], cl.indices[2],
                                cl.weights[0], cl.weights[1], cl.weights[2]);
  }

  Vector3f getVort(float x, float y, float z) {
    CellLerp cl = this->g->getVoxelLerp(x, y, z);
    if (this->vortSol != nullptr) {
      return this->vortSol->getValLerp(cl.indices[0], cl.indices[1], cl.indices[2],
                                              cl.weights[0], cl.weights[1], cl.weights[2]);
    }
    Matrix3f jac = getJac(x, y, z);
    return calcVort(jac);
  }

  float getVortMag(float x, float y, float z) {
    CellLerp cl = this->g->getVoxelLerp(x, y, z);
    if (this->vortMagSol != nullptr) {
      return this->vortMagSol->getValLerp(cl.indices[0], cl.indices[1], cl.indices[2],
                                              cl.weights[0], cl.weights[1], cl.weights[2]);
    }
    Vector3f vort = getVort(x, y, z);
    return vort.norm();
  }


};



template <typename T=float>
class HeartEquation: public ScalarField<T> {
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