#ifndef SVK_RENDERER_TRANSFERFUNCTION_H
#define SVK_RENDERER_TRANSFERFUNCTION_H

#include "PieceWiseFunction.h"
#include "common/numerical.h"
#include <glm/glm.hpp>
// #include <Eigen/Dense>

class TransferFunction {
  private:
    PieceWiseFunction tfRGBA;
    PieceWiseFunction tfRGB;
    PieceWiseFunction tfA;
  public:
    TransferFunction(): tfRGB(), tfA() {}

    // SET DEFAULT MIN MAX VALUE
    void setDefaultMinMaxRGB(glm::vec3 minval, glm::vec3 maxval) {
      this->tfRGB.setDefaultMinMaxValVec3(minval, maxval);
    }
    void setDefaultMinMaxOpacity(float minval, float maxval) {
      this->tfA.setDefaultMinMaxVal(minval, maxval);
    }

    // GET VALUE
    glm::vec4 getRGBA(double x) {
      return tfRGBA.getValueVec4(x);
    }

    glm::vec3 getRGB(double x) {
      return tfRGB.getValueVec3(x);
    }
    
    double getOpacity(double x) {
      return tfA.getValue(x);
    }

    // ADD CONTROL POINT
    void addRGBAPoint(double x, double r, double g, double b, double a) {
      this->tfRGBA.addPoint(x, glm::vec4(r, g, b, a));
    }
    void addRGBPoint(double x, double r, double g, double b) {
      this->tfRGB.addPoint(x, glm::vec3(r, g, b));
    }

    void addOpacityPoint(double x, double opacity) {
      this->tfA.addPoint(x, opacity);
    }

    // REMOVE CONTROL POINT
    void removeRGBAPoint(double x) {
      this->tfRGBA.removePoint(x);
    }
    void removeRGBPoint(double x) {
      this->tfRGB.removePoint(x);
    }
    void removeOpacityPoint(double x) {
      this->tfA.removePoint(x);
    }

    // GET CONTROL POINT RANGE
    const double* getRGBAPointRange() {
      return this->tfRGBA.range;
    }
    const double* getRGBPointRange() {
      return this->tfRGB.range;
    }
    const double* getOpacityPointRange() {
      return this->tfA.range;
    }


};

#endif