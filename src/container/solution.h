#ifndef SVK_CONTRAINER_SOLUTION_H
#define SVK_CONTRAINER_SOLUTION_H

// #define _GLIBCXX_USE_CXX11_ABI 0
#include "common/util.h"
#include "common/numerical.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cerrno>
#include <vector>

#include <Eigen/Dense>
using namespace Eigen;

// #include <numeric>
// #include <functional>

template <typename T=float>
class Solution {
  private:
    
    // update index strides. called when dimension is set.
    void updateStrides() {
      int cumulated = 1;
      int i;
      // stride[-1] = 1
      // stride[k] = product(dims[k+1:])
      for (i=this->dims.size()-1; i > 0; i--) {
        // std::cout << "bbbbbbb stride " << i << this->strides[i] << "\n";
        this->strides[i] = cumulated;
        // std::cout << "stride " << i  <<": "<< cumulated << "\n";
        cumulated *= this->dims[i];
      }
      this->strides[0] = cumulated;
      // std::cout << "stride " << i << ": "<<cumulated << "\n";
    }

  public:
    std::vector<T> data;
    int length;
    // int precision;
    std::vector<int> dims;
    std::vector<int> strides;
    // ************************************************************************
    // constructor
    // default: init data to NULL
    Solution() {}

    // Construction 1: specify # of dims and then fill in the dimension length
    Solution(int numdims) {
      this->dims = std::vector<int>(numdims, 0);
      this->strides = std::vector<int>(numdims, 0);
      this->length = 0;
    }
    // Construction 2: specify the 3 dimensions' lengths
    Solution(int xdim, int ydim, int zdim) { this->init3D(xdim, ydim, zdim); }
    // Construction 3: specify the 2 dimensions' lengths
    Solution(int xdim, int ydim) { this->init2D(xdim, ydim); }

    void init3D(int xdim, int ydim, int zdim) {
      this->dims = std::vector<int>(3, 0);
      this->strides = std::vector<int>(3, 0);

      this->setDimLen(0, xdim);
      this->setDimLen(1, ydim);
      this->setDimLen(2, zdim);
      this->initData();
    }
    void init2D(int xdim, int ydim) {
      this->dims = std::vector<int>(2, 0);
      this->strides = std::vector<int>(2, 0);

      this->setDimLen(0, xdim);
      this->setDimLen(1, ydim);
      this->initData();
    }

    // ************************************************************************
    // solution properties
    std::vector<int> getDimensions() {
      return this->dims;
    }
    int getLen() { return this->data.size(); }
    int getDimLen(int idim) { return this->dims[idim]; }
    void setDimLen(int idim, int dlen) {
      this->dims[idim] = dlen;
      this->length = product(dims);
      this->updateStrides();
    }

    // for a vector solution, swap the data in two dimensions
    void swapAxes(int idim, int jdim, int kdim) {
      std::vector<int> tmpStrides(this->strides);
      this->strides[0] = tmpStrides[idim];
      this->strides[1] = tmpStrides[jdim];
      this->strides[2] = tmpStrides[kdim];
    }

    void swapVecDim(int idim, int jdim, int kdim) {
      // std::cout << this->data[0] << "\nvs\n";
      for (int i = 0; i < this->length; i++) {
        T thisvec = this->data[i];
        T tmpvec = thisvec;
        thisvec(0) = tmpvec(idim);
        thisvec(1) = tmpvec(jdim);
        thisvec(2) = tmpvec(kdim);
        this->data[i] = thisvec;
      }
      // std::cout << this->data[0] << "\n";
    }
    // std::vector<T> range() {}

    // ************************************************************************
    // data I/O
    void load(const std::string& fpath, bool verbose=false) {
      std::ifstream fdata(fpath, std::ios::binary);
      if (fdata.fail()) {
        printf("File reading failed (%s)", fpath);
      }
      if (this->data.empty()) {
        this->initData();
        std::cout << "initialized data upon reading." << std::endl;
      }
      fdata.read(reinterpret_cast<char*>(this->data.data()), sizeof(T)*this->length);
      fdata.close();
      if (verbose)
        std::cout << "Read " << this->length << " of " << sizeof(T) << "-byte data." << std::endl;
    }

    // initialize the Solution from vec
    void fromVec(const std::string& fpath, bool verbose=false) {

      std::ifstream fdata(fpath, std::ios::binary);
      if (fdata.fail()) {
        printf("File reading failed (%s): ", fpath.c_str());
        std::cerr << strerror(errno) << "\n";
        return;
      }

      std::vector<int> dims(3);
      fdata.read(reinterpret_cast<char*>(dims.data()), sizeof(int)*3);
      printf("%i, %i, %i, dims\n", dims[0], dims[1], dims[2]);
      this->init3D(dims[0], dims[1], dims[2]);
      fdata.read(reinterpret_cast<char*>(this->data.data()), sizeof(T)*this->length);
      if (verbose) {
          printf("Solution::fromVec complete: Solution read %i (%ix%ix%i) 3D float vector from \"%s\"\n",
          this->length, dims[0], dims[1], dims[2], fpath.c_str());
      }
      fdata.close();
    }

    // dump raw data to fpath
    void save(const std::string& fpath, bool verbose=true) {
      std::ofstream fdata(fpath, std::ios::binary);
      // do nothing if data hasn't been loaded
      if (this->data.empty()) {
        std::cout << "Save failed. No data loaded." << std::endl;
        fdata.close();
        return;
      }

      fdata.write(reinterpret_cast<char*>(this->data.data()), sizeof(T)*this->length);
      fdata.close();
      if (verbose)
        std::cout << "Saved " << this->length << " of " << sizeof(T) << "-byte data to " << fpath << std::endl;
    }
    
    auto castPrecision(float val);
    

    // ************************************************************************
    // data & value

    void initData() {
      if (this->length == 0) {
        std::cerr << "Set all dimension lengths before initializing data!\n";
        return;
      }
      // // free memory if data allocated before
      // if (data != nullptr) {
      //   delete[] data;
      //   data = 0;
      // }
      this->data.clear();
      this->data.resize(this->length);
    }
    // convert x,y,z index to the index of 1d array
    int index(int xi, int yi, int zi) {
      return xi*this->strides[0] + yi*this->strides[1] + zi*this->strides[2];
    }
    int index(int xi, int yi) {
      return xi*this->strides[0] + yi*this->strides[1];
    }

    // set the value at the indices
    void setVal(int xi, int yi, int zi, T val) {
      int i = this->index(xi, yi, zi);
      this->data[i] = val;
      // std::cout << "r u empty? " << this->data.size() << "readlly?? " << this->data[i] << "\n";
    }

    void setVal(int xi, int yi, T val) {
      int i = this->index(xi, yi);
      this->data[i] = val;
      // std::cout << "r u empty? " << this->data.size() << "readlly?? " << this->data[i] << "\n";
    }

    // get the value at the indices
    T getVal(int xi, int yi, int zi) {
      // reflect padding: handle out-of-bound dimensions (for central different)
      if (xi == -1) xi += 2;
      else if (xi == this->getDimLen(0)) xi -= 2;

      if (yi == -1) yi += 2;
      else if (yi == this->getDimLen(1)) yi -= 2;

      if (zi == -1) zi += 2;
      else if (zi == this->getDimLen(2)) zi -= 2;

      int i = this->index(xi, yi, zi);
      // printf("index %i - \n", i);
      // std::cout << this->data[i] << "\n";
      return this->data[i];
    }

    T getVal(int xi, int yi) {
      // reflect padding: handle out-of-bound dimensions (for central different)
      if (xi == -1) xi += 2;
      else if (xi == this->getDimLen(0)) xi -= 2;

      if (yi == -1) yi += 2;
      else if (yi == this->getDimLen(1)) yi -= 2;

      int i = this->index(xi, yi);
      return this->data[i];
    }

    // Central Difference gradient approximation at the indices
    glm::vec3 getGrad(int xi, int yi, int zi) {
      glm::vec3 g;
      g[0] = (this->getVal(xi+1, yi, zi) - this->getVal(xi-1, yi, zi)) / 2.f;
      g[1] = (this->getVal(xi, yi+1, zi) - this->getVal(xi, yi-1, zi)) / 2.f;
      g[2] = (this->getVal(xi, yi, zi+1) - this->getVal(xi, yi, zi-1)) / 2.f;
      return g;
    }

    // Central Difference gradient approximation at the indices
    Matrix3Xf getJac(int xi, int yi, int zi) {
      Matrix3Xf g(3, 3);
      g.row(0) = (this->getVal(xi+1, yi, zi) - this->getVal(xi-1, yi, zi)) / 2.f;
      g.row(1) = (this->getVal(xi, yi+1, zi) - this->getVal(xi, yi-1, zi)) / 2.f;
      g.row(2) = (this->getVal(xi, yi, zi+1) - this->getVal(xi, yi, zi-1)) / 2.f;
      return g;
    }

    //		    6________7
    //		   /|       /|
    //		  / |      / |
    //		4/_______5/  |
    //		|  2|___ |___|3
    //		|  /     |  /
    //		| /      | /
    //		|/_______|/
    //  **0**      1
    // (xi, yi, zi) = 0;
    // given smallest corner indices of a cell and weights, return trilerp value interpolant for this cell
    T getValLerp(int xi, int yi, int zi, float alpha, float beta, float gamma) {
      T v0 = this->getVal(xi, yi, zi);
      T v1 = this->getVal(xi+1, yi, zi);
      T v2 = this->getVal(xi, yi+1, zi);
      T v3 = this->getVal(xi+1, yi+1, zi);
      T v4 = this->getVal(xi, yi, zi+1);
      T v5 = this->getVal(xi+1, yi, zi+1);
      T v6 = this->getVal(xi, yi+1, zi+1);
      T v7 = this->getVal(xi+1, yi+1, zi+1);
      // std::cout << "TRILERP: " << v0 << " " << v1<<" " <<v2<<" " <<v3<<" " <<v4<<" " <<v5<<" " <<v6<<" " <<v7<< "-" << alpha << " " << beta << " " << gamma << "\n";
      return trilerp(v0, v1, v2, v3, v4, v5, v6, v7, alpha, beta, gamma);
    }

    T getValLerp(int xi, int yi, float alpha, float beta) {
      T v0 = this->getVal(xi, yi);
      T v1 = this->getVal(xi+1, yi);
      T v2 = this->getVal(xi, yi+1);
      T v3 = this->getVal(xi+1, yi+1);
      // std::cout << "TRILERP: " << v0 << " " << v1<<" " <<v2<<" " <<v3<<" " <<v4<<" " <<v5<<" " <<v6<<" " <<v7<< "-" << alpha << " " << beta << " " << gamma << "\n";
      return bilerp(v0, v1, v2, v3, alpha, beta);
    }

    //		    6________7
    //		   /|       /|
    //		  / |      / |
    //		4/_______5/  |
    //		|  2|___ |___|3
    //		|  /     |  /
    //		| /      | /
    //		|/_______|/
    //  **0**      1
    // (xi, yi, zi) = 0;
    // given smallest corner indices of a cell and weights, return trilerp gradient interpolant for this cell
    glm::vec3 getGradLerp(int xi, int yi, int zi, float alpha, float beta, float gamma) {
      glm::vec3 g0 = this->getGrad(xi, yi, zi);
      glm::vec3 g1 = this->getGrad(xi+1, yi, zi);
      glm::vec3 g2 = this->getGrad(xi, yi+1, zi);
      glm::vec3 g3 = this->getGrad(xi+1, yi+1, zi);
      glm::vec3 g4 = this->getGrad(xi, yi, zi+1);
      glm::vec3 g5 = this->getGrad(xi+1, yi, zi+1);
      glm::vec3 g6 = this->getGrad(xi, yi+1, zi+1);
      glm::vec3 g7 = this->getGrad(xi+1, yi+1, zi+1);
      glm::vec3 p;
      p = trilerp(g0, g1, g2, g3, g4, g5, g6, g7, alpha, beta, gamma);
      // p[0] = trilerp(g0[0], g1[0], g2[0], g3[0], g4[0], g5[0], g6[0], g7[0], alpha, beta, gamma);
      // p[1] = trilerp(g0[1], g1[1], g2[1], g3[1], g4[1], g5[1], g6[1], g7[1], alpha, beta, gamma);
      // p[2] = trilerp(g0[2], g1[2], g2[2], g3[2], g4[2], g5[2], g6[2], g7[2], alpha, beta, gamma);
      return p;
    }

    Matrix3f getJacLerp(int xi, int yi, int zi, float alpha, float beta, float gamma) {
      Matrix3f g0 = this->getJac(xi, yi, zi);
      Matrix3f g1 = this->getJac(xi+1, yi, zi);
      Matrix3f g2 = this->getJac(xi, yi+1, zi);
      Matrix3f g3 = this->getJac(xi+1, yi+1, zi);
      Matrix3f g4 = this->getJac(xi, yi, zi+1);
      Matrix3f g5 = this->getJac(xi+1, yi, zi+1);
      Matrix3f g6 = this->getJac(xi, yi+1, zi+1);
      Matrix3f g7 = this->getJac(xi+1, yi+1, zi+1);
      Matrix3f p;
      p = trilerp(g0, g1, g2, g3, g4, g5, g6, g7, alpha, beta, gamma);
      // p[0] = trilerp(g0[0], g1[0], g2[0], g3[0], g4[0], g5[0], g6[0], g7[0], alpha, beta, gamma);
      // p[1] = trilerp(g0[1], g1[1], g2[1], g3[1], g4[1], g5[1], g6[1], g7[1], alpha, beta, gamma);
      // p[2] = trilerp(g0[2], g1[2], g2[2], g3[2], g4[2], g5[2], g6[2], g7[2], alpha, beta, gamma);
      return p;
    }

    void setData(const std::vector<T> &data) {
      this->data = data;
    }

    const std::vector<T>& getData() {
      return this->data;
    }
};

#endif