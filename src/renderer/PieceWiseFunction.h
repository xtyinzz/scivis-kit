#ifndef SVK_RENDERER_PIECEWISEFUNCTION_H
#define SVK_RENDERER_PIECEWISEFUNCTION_H

#include "common/numerical.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <glm/glm.hpp>
// #include <Eigen/Dense>
/*
reference: vtkPieceWiseFunction
*/

class PieceWiseFunctionNode {
  public:
    double x;
    double y;
    glm::vec3 v3;
    glm::vec4 v4;
    PieceWiseFunctionNode() {}
    PieceWiseFunctionNode(double x, double y): x(x), y(y) {}
    PieceWiseFunctionNode(double x, glm::vec3 vec): x(x), v3(vec) {}
    PieceWiseFunctionNode(double x, glm::vec4 vec): x(x), v4(vec) {}


    bool operator== (const PieceWiseFunctionNode& n) {
      return this->x == n.x;
    }

    bool operator!= (const PieceWiseFunctionNode& n) {
      return this->x != n.x;
    }

    bool operator< (const PieceWiseFunctionNode& n) {
      return this->x < n.x;
    }

    bool operator<= (const PieceWiseFunctionNode& n) {
      return this->x <= n.x;
    }

    bool operator> (const PieceWiseFunctionNode& n) {
      return this->x > n.x;
    }

    bool operator>= (const PieceWiseFunctionNode& n) {
      return this->x >= n.x;
    }
};


class PieceWiseFunction {
  private:
    static bool compare_node_pointer(const PieceWiseFunctionNode *n1, const PieceWiseFunctionNode *n2) {
      return (n1->x < n2->x);
    }

    

  public:
    enum ValueType {
      SCALAR=0, VECTOR=1
    };

    enum InterpMethod {
      LINEAR=0, STEP=1
    };


    float default_minval = 0.f;
    float default_maxval = 0.f;
    glm::vec3 default_minvalVec3 = glm::vec3(0.f);
    glm::vec3 default_maxvalVec3 = glm::vec3(0.f);
    glm::vec4 default_minvalVec4 = glm::vec4(0.f);
    glm::vec4 default_maxvalVec4 = glm::vec4(0.f);

    std::vector<PieceWiseFunctionNode*> nodes;
    double range[2] = {0., 0.};
    ValueType type = this->SCALAR;

    PieceWiseFunction() {}
    ~PieceWiseFunction() {
      for (size_t i = 0; i < this->nodes.size(); i++) {
        delete this->nodes[i];
      }
    }


    void setDefaultMinMaxVal(float minval, float maxval) {
      this->default_minval = minval;
      this->default_maxval = maxval;
    }
    void setDefaultMinMaxValVec3(glm::vec3 minval, glm::vec3 maxval) {
      this->default_minvalVec3 = minval;
      this->default_maxvalVec3 = maxval;
    }
    void setDefaultMinMaxValVec4(glm::vec4 minval, glm::vec4 maxval) {
      this->default_minvalVec3 = minval;
      this->default_maxvalVec4 = maxval;
    }

    void setValueType(ValueType type) {
      this->type = type;
    }

    double getValue(double x) {
      // boundary case: x outside of range: return 0
      // std::cout << x <<" inrange?: " <<(int)(this->isInRange(x)) << "\n";
      if (this->isBelowMin(x)) {
        return this->default_minval;
      } else if (this->isAboveMax(x)) {
        return this->default_maxval;
      }

      // loop thru and lerp
      size_t i;
      double value = 0.;
      for (i = 0; i < this->nodes.size()-1; i++) {
        // std::cout << i << " node i=" << this->nodes[i]->x << ", i+1 = " << this->nodes[i+1]->x << std::endl;
        if ((this->nodes[i]->x <= x) && (this->nodes[i+1]->x >= x)) {
          value = lerp(x, this->nodes[i]->x, this->nodes[i+1]->x, this->nodes[i]->y, this->nodes[i+1]->y);
          // std::cout << "found value = " << value << std::endl;
          break;
        }
      }
      return value;
    }

    glm::vec3 getValueVec3(double x) {
      // boundary case: x outside of range: return 0
      // std::cout << x <<" inrange?: " <<(int)(this->isInRange(x)) << "\n";
      if (this->isBelowMin(x)) {
        return glm::vec3(this->default_minval);
      } else if (this->isAboveMax(x)) {
        return glm::vec3(this->default_maxval);
      }
      // loop thru and lerp
      size_t i;
      glm::vec3 value;
      for (i = 0; i < this->nodes.size()-1; i++) {
        // std::cout << i << " node i=" << this->nodes[i]->x << ", i+1 = " << this->nodes[i+1]->x << std::endl;
        if ((this->nodes[i]->x <= x) && (this->nodes[i+1]->x >= x)) {
          value = lerpGLM(x, this->nodes[i]->x, this->nodes[i+1]->x, this->nodes[i]->v3, this->nodes[i+1]->v3);
          // std::cout << "found value = " << value << std::endl;
          break;
        }
      }
      return value;
    }

    glm::vec4 getValueVec4(double x) {
      // boundary case: x outside of range: return 0
      // std::cout << x <<" inrange?: " <<(int)(this->isInRange(x)) << "\n";
      if (this->isBelowMin(x)) {
        return glm::vec4(this->default_minval);
      } else if (this->isAboveMax(x)) {
        return glm::vec4(this->default_maxval);
      }
      // loop thru and lerp
      size_t i;
      glm::vec4 value;
      for (i = 0; i < this->nodes.size()-1; i++) {
        // std::cout << i << " node i=" << this->nodes[i]->x << ", i+1 = " << this->nodes[i+1]->x << std::endl;
        if ((this->nodes[i]->x <= x) && (this->nodes[i+1]->x >= x)) {
          value = lerpGLM(x, this->nodes[i]->x, this->nodes[i+1]->x, this->nodes[i]->v4, this->nodes[i+1]->v4);
          // std::cout << "found value = " << value << std::endl;
          break;
        }
      }
      return value;
    }

    bool isInRange(double x) {
      return !((this->range[0] > x) || (this->range[1] < x));
    }
    bool isBelowMin(double x) {
      return this->range[0] > x;
    }
    bool isAboveMax(double x) {
      return this->range[1] < x;
    }

    void printPoints() {
      std::cout << "Range: [" << this->range[0] << ", " << this->range[1] << "]\n";
      std::cout << "[";
      for (size_t i = 0; i < this->nodes.size(); i++) {
        std::cout << " (" << this->nodes[i]->x << ", " << this->nodes[i]->y << ") ";
      }
      std::cout << "]\n";
    }

    // Add contorl point, same previously will be removed

    void addPoint(double x, glm::vec4 y) {
      this->removePoint(x);
      PieceWiseFunctionNode* node = new PieceWiseFunctionNode(x, y);
      this->nodes.push_back(node);
      this->SortAndUpdateRange();
    }

    
    void addPoint(double x, glm::vec3 y) {
      this->removePoint(x);
      PieceWiseFunctionNode* node = new PieceWiseFunctionNode(x, y);
      this->nodes.push_back(node);
      this->SortAndUpdateRange();
    }

    void addPoint(double x, double y) {
      this->removePoint(x);
      PieceWiseFunctionNode* node = new PieceWiseFunctionNode(x, y);
      this->nodes.push_back(node);
      this->SortAndUpdateRange();
    }

    void SortAndUpdateRange() {
      std::stable_sort(
      this->nodes.begin(), this->nodes.end(), PieceWiseFunction::compare_node_pointer);
      this->updateRange();
    }

    void updateRange() {
      size_t size = this->nodes.size();
      // int size = static_cast<int>(this->nodes.size());
      this->range[0] = this->nodes[0]->x;
      this->range[1] = this->nodes[size-1]->x;
    }

    bool removePointByIndex(size_t i) {
      if (i >= this->nodes.size()) return false;

      delete this->nodes[i];
      this->nodes.erase(this->nodes.begin() + i);
      return true;
    }

    int removePoint(double x) {
      size_t i;
      for (i = 0; i < this->nodes.size(); i++) {
        if (this->nodes[i]->x == x) {
          break;
        }
      }
      // If the node DNE return -1
      if (i == this->nodes.size())
      {
        return -1;
      }
      this->removePointByIndex(i);
      this->updateRange();
      return static_cast<int>(i);
    }

    const std::vector<PieceWiseFunctionNode*> * getPoints() {
      return &nodes;
    }

    void adjustRange(double range[2]) {
      this->range[0] = range[0];
      this->range[1] = range[1];
    }

    void adjustRange(double rangemin, double rangemax) {
      this->range[0] = rangemin;
      this->range[1] = rangemax;
    }

    const double* getRange() {
      return this->range;
    }
};

#endif