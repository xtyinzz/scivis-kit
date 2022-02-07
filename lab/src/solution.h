#ifndef SOLUTION_H
#define SOLUTION_H

// #define _GLIBCXX_USE_CXX11_ABI 0
#include "util.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
// #include <numeric>
// #include <functional>

template <typename T=float>
class Solution {
  private:
    
    // update index strides. called when dimension is set.
    void updateStrides() {
      size_t cumulated = 1;
      unsigned long long i;
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
    std::vector<unsigned long long > strides;
    // ************************************************************************
    // constructor
    // default: init data to NULL
    Solution() {}

    // Construction 1: specify # of dims and then fill in the dimension length
    Solution(int numdims) {
      this->dims = std::vector<int>(numdims, 0);
      this->strides = std::vector<unsigned long long>(numdims, 0);
      this->length = 0;
    }
    // Construction 2: specify the 3 dimensions' lengths
    Solution(int xdim, int ydim, int zdim) {
      this->dims = std::vector<int>(3, 0);
      this->strides = std::vector<unsigned long long>(3, 0);

      this->setDimLen(0, xdim);
      this->setDimLen(1, ydim);
      this->setDimLen(2, zdim);
      // this->initData();
    }
    // ~Solution() { delete[] data; }


    // ************************************************************************
    // solution properties
    int getDimLen(int idim) { return this->dims[idim]; }
    void setDimLen(int idim, int dlen) {
      this->dims[idim] = dlen;
      this->length = product(dims);
      this->updateStrides();
    }
    // std::vector<T> range() {}

    // ************************************************************************
    // data I/O
    void load(const std::string& fpath) {
      std::ifstream fdata(fpath, std::ios::binary);
      this->initData();

      if (this->data.empty()) {
        this->initData();
        std::cout << "initialized data upon reading." << std::endl;
      }
      fdata.read(reinterpret_cast<char*>(this->data.data()), sizeof(T)*this->length);
      fdata.close();
      std::cout << "Read " << this->length << " of " << sizeof(T) << "-byte data." << std::endl;
    }
    void save(const std::string& fpath) {
      std::ofstream fdata(fpath, std::ios::binary);
      // do nothing if data hasn't been loaded
      if (this->data.empty()) {
        std::cout << "Save failed. No data loaded." << std::endl;
        fdata.close();
        return;
      }

      fdata.write(reinterpret_cast<char*>(this->data.data()), sizeof(T)*this->length);
      fdata.close();
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
      this->data.resize(this->length);
    }
    // convert x,y,z index to the index of 1d array
    int index(int xi, int yi, int zi) {
      return xi*this->strides[0] + yi*this->strides[1] + zi*this->strides[2];
    }

    // set the value at the indices
    void setVal(int xi, int yi, int zi, T val) {
      int i = this->index(xi, yi, zi);
      this->data[i] = val;
      // std::cout << "r u empty? " << this->data.size() << "readlly?? " << this->data[i] << "\n";
    }

    // get the value at the indices
    T val(int xi, int yi, int zi) {
      int i = this->index(xi, yi, zi);
      return this->data[i];
    }

    // Central Difference gradient approximation at the indices
    std::vector<T> grad(int xi, int yi, int zi) {
      std::vector<T> g(3, 0);
      g[0] = (this->val(xi+1, yi, zi) - this->val(xi-1, yi, zi)) / 2;
      g[1] = (this->val(xi, yi+1, zi) - this->val(xi, yi-1, zi)) / 2;
      g[2] = (this->val(xi, yi, zi+1) - this->val(xi, yi, zi-1)) / 2;
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
    T val_trilerp(int xi, int yi, int zi, float alpha, float beta, float gamma) {
      T v0 = this->val(xi, yi, zi);
      T v1 = this->val(xi+1, yi, zi);
      T v2 = this->val(xi, yi+1, zi);
      T v3 = this->val(xi+1, yi+1, zi);
      T v4 = this->val(xi, yi, zi+1);
      T v5 = this->val(xi+1, yi, zi+1);
      T v6 = this->val(xi, yi+1, zi+1);
      T v7 = this->val(xi+1, yi+1, zi+1);
      // std::cout << "TRILERP: " << v0 << " " << v1<<" " <<v2<<" " <<v3<<" " <<v4<<" " <<v5<<" " <<v6<<" " <<v7<< "-" << alpha << " " << beta << " " << gamma << "\n";
      return trilerp(v0, v1, v2, v3, v4, v5, v6, v7, alpha, beta, gamma);
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
    std::vector<T> grad_trilerp(int xi, int yi, int zi, float alpha, float beta, float gamma) {
      std::vector<T> g0 = this->grad(xi, yi, zi);
      std::vector<T> g1 = this->grad(xi+1, yi, zi);
      std::vector<T> g2 = this->grad(xi, yi+1, zi);
      std::vector<T> g3 = this->grad(xi+1, yi+1, zi);
      std::vector<T> g4 = this->grad(xi, yi, zi+1);
      std::vector<T> g5 = this->grad(xi+1, yi, zi+1);
      std::vector<T> g6 = this->grad(xi, yi+1, zi+1);
      std::vector<T> g7 = this->grad(xi+1, yi+1, zi+1);
      std::vector<T> p(3, 0);
      for (int i = 0; i < 3; i++) {
        p[i] = trilerp(g0[i], g1[i], g2[i], g3[i], g4[i], g5[i], g6[i], g7[i], alpha, beta, gamma);
      }
      return p;
    }
};

#endif