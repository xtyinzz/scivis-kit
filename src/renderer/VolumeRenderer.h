#ifndef SVK_RENDERER_VOLUMERENDERER_H
#define SVK_RENDERER_VOLUMERENDERER_H

#include "VolumeRenderer.h"
#include "TransferFunction.h"
#include "container/field.h"
#include "container/solution.h"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#include <Eigen/Dense>
// this is added after CImg.h include
// to prevent redefinition of Success: conflict b/t CImg and Eigen
#ifdef Success
  #undef Success
#endif

// this used to prevent redefinition of Success: conflict b/t CImg and Eigen
// #ifdef Success
//   #undef Success
// #endif

class Camera {
  public:
    double r, azimuth, elevation;
    glm::vec3 center;

    Camera() {}

    void setAzimuth(double deg);
    void setElevation(double deg);
    void setCenter(glm::vec3 coord);
};


class Image: public Solution<glm::vec4> {
  private:

  public:
    enum Format {
      GRAY=0, RGB=1, RGBA=2
    };
    // default initialize a 2D solution
    Image(): Solution(2) {}
    // intialize with width and height
    Image(size_t width, size_t height): Solution(width, height) {}
    //
    Format format = this->RGBA;

    void setSize(size_t width, size_t height) {
      this->Solution::setDimLen(0, width);
      this->Solution::setDimLen(1, height);
      this->Solution::initData();
    }
    
    size_t getWidth() {
      return this->Solution::getDimLen(0);
    }
    size_t getHeight() {
      return this->Solution::getDimLen(1);
    }
    std::vector<int> getSize() {
      return this->Solution::getDimensions();
    }

    void setOpacity(size_t x, size_t y, double opacity) {
      this->Solution::setVal(x, y, glm::vec4(0.,0.,0.,opacity));
    }
    void setRGBA(size_t x, size_t y, double r, double g, double b, double opacity) {
      this->Solution::setVal(x, y, glm::vec4(r,g,b,opacity));
    }
    void setRGB(size_t x, size_t y, double r, double g, double b) {
      this->Solution::setVal(x, y, glm::vec4(r,g,b,1.));
    }

    double getOpacity(size_t x, size_t y) {
      return this->Solution::getVal(x, y).a;
    }
    glm::vec4 getRGBA(size_t x, size_t y) {
      return this->Solution::getVal(x, y);
    }
    std::vector<glm::vec4> getImageRGBA() {
      return std::vector<glm::vec4>();
    }

    void writePNG(const std::string& fpath, int num_channel=3) {
      Solution<unsigned char> img8bit(this->getWidth(), this->getHeight(), num_channel);

      for (size_t i = 0; i < this->dims[0]; i++) {
        for (size_t j = 0; j < this->dims[1]; j++) {
          glm::vec4 rgba_float = this->getRGBA(i, j);
          // printVec(rgba_float);
          // clamp at min of 0 and max of 255
          for (size_t c = 0; c < num_channel; c++) {
            unsigned char cval_8bit = std::clamp(
              (int)std::round(rgba_float[c]*255),
              0,
              255
            );
            img8bit.setVal(i, j, c, cval_8bit);
            // if (rgba_float[c] > 1. || rgba_float[c] < 0.) {
            //   std::cout << (int)cval_8bit << " and " << c << " = ";
            //   printVec(rgba_float);
            // } else {
            //   // std::cout << rgba_float[c] << " and " << (int)cval_8bit << "\n";
            // }
          }
          // printf(" Pixel (%-3zu, %-3zu): (%-3u, %-3u, %-3u)\n", i, j, img8bit.getVal(i,j,0), img8bit.getVal(i,j,1), img8bit.getVal(i,j,2));
        }
      }
      // printVec(std::max_element(this->data.begin(), this->data.end()));
      // std::cout << (int)std::max_element(img8bit.getData()->begin(), img8bit.getData()->end()) << "\n";
      const unsigned char *imgArray = img8bit.getData();
      stbi_write_png(fpath.c_str(), this->dims[0], this->dims[1], num_channel, imgArray, this->dims[0] * num_channel);
      std::cout << "image written to " << fpath << std::endl;
    }

    //
    // round
    //int n = (int)(sqrt(3)+0.5f)
};

class Light {
  public:
    glm::vec3 position;
    float amb;
    float dif;
    float spec;
    glm::vec3 cAmb;
    glm::vec3 cDif;
    glm::vec3 cSpec;

    Light(): cAmb(1.f), cDif(1.f), cSpec(1.f) {}
};

class VolumeRenderer {
  private:
    
  public:
    enum Shading {
      SHADING_NONE=0, SHADING_PHONG=1
    };

    Field<float> *field = nullptr;
    Field<glm::vec3> *gradField = nullptr;
    TransferFunction *tf = nullptr;
    Image img;
    Shading shading = this->SHADING_NONE;
    Light light;

    
    VolumeRenderer() {}
    Light getLight(glm::vec3 light) {
      return this->light;
    }
    void setLight(glm::vec3 position, float amb, float dif, float spec) {
      this->light.position = position;
      this->light.amb = amb;
      this->light.dif = dif;
      this->light.spec = spec;
    }
    void setLightColor(glm::vec3 cAmb, glm::vec3 cDif, glm::vec3 cSpec) {
      this->light.cAmb = cAmb;
      this->light.cDif = cDif;
      this->light.cSpec = cSpec;
    }

    VolumeRenderer::Shading getShading() {
      return this->shading;
    }

    void setShading(VolumeRenderer::Shading shading) {
      this->shading = shading;
    }

    glm::vec3 getPhongColor(glm::vec3 color, glm::vec3 eye2sample, glm::vec3 normal) {
      glm::vec3 sample2eye = glm::normalize(-eye2sample);
      glm::vec3 sample2light = glm::normalize(this->light.position - eye2sample);

      float nl = glm::max(glm::dot(normal, sample2light), 0.f);
      glm::vec3 r = glm::normalize(2.f * nl * normal - sample2light);
      float rv = glm::max(glm::dot(r, sample2eye), 0.f);

      glm::vec3 ambient = this->light.amb * this->light.cAmb; 
      glm::vec3 diffuse = this->light.dif * this->light.cDif * nl;
      glm::vec3 specular = this->light.spec * this->light.cSpec * rv;
      // printVec(normal);
      // printVec(sample2light);
      // printVec(sample2eye);
      // std::cout <<nl << " and " << rv <<"\n";

      return color*(ambient + diffuse + specular);
    }

    void setImageFormat(Image::Format format) {
      this->img.format = format;
    }

    void setTransferFunction(TransferFunction *tf) {
      this->tf = tf;
    }

    void setField(Field<float> *field) {
      this->field = field;
    }

    void setGradField(Field<glm::vec3> *gradField) {
      this->gradField = gradField;
    }

    void setImageDimension(int width, int height) {
      this->img.setSize(width, height);
    }
    std::vector<int> getImageDimension() {
      return this->img.getSize();
    }

    void writePNG(const std::string& fpath, int num_channel=3) {
      this->img.writePNG(fpath, num_channel);
    }
    // camera model?
    // glm::vec3 rayCorner
    void render(glm::vec3 rayStep, int numDepths, int planeIdx[]) {
      std::vector<std::vector<double>> fieldExtents = {
        this->field->getDimExtent(0),
        this->field->getDimExtent(1),
        this->field->getDimExtent(2),
      };

      double ustep = (fieldExtents[planeIdx[0]][1] - fieldExtents[planeIdx[0]][0]) / (img.getWidth()-1);
      double vstep = (fieldExtents[planeIdx[1]][1] - fieldExtents[planeIdx[1]][0]) / (img.getHeight()-1);

      double xmin = fieldExtents[0][0];
      double ymin = fieldExtents[1][0];
      double zmin = fieldExtents[2][0];
      glm::vec3 rayCorner(xmin,ymin,zmin);
      // for every pixel/ray
      std::cout << "ray starts at corner";
      printVec(rayCorner);
      #pragma omp parallel for collapse(2)
      for (size_t u=0; u < this->img.getWidth(); u++) {
        for (size_t v=0; v < this->img.getHeight(); v++) {

          glm::vec3 ray(rayCorner);
          ray[planeIdx[0]] += u*ustep;
          ray[planeIdx[1]] += v*vstep;

          glm::vec3 cIn(0.f);
          glm::vec3 cOut(0.f);
          float opacityIn = 0;
          float opacityOut = 0;
          for (int depth = 0; depth < numDepths; depth++) {
            // get current value, rgba
            if (!this->field->isBounded(ray.x, ray.y, ray.z)) break;
            double intensity = this->field->getVal(ray.x, ray.y, ray.z);
            glm::vec3 rgb = this->tf->getRGB(intensity);
            float opacity = this->tf->getOpacity(intensity);

            // std::cout << intensity << " -> rgba ";
            // printVec(glm::vec4(cOut, opacityOut));

            if (shading == this->SHADING_PHONG) { 
              // std::vector<float> tmpg = this->field->getGrad(ray.x, ray.y, ray.z);
              // glm::vec3 gradient(tmpg[0], tmpg[1], tmpg[2]);

              glm::vec3 gradient = this->field->getGrad(ray.x, ray.y, ray.z);
              glm::vec3 normal(0.f);
              if (glm::length(gradient) > 0.f) {
                normal = glm::normalize(gradient);
              }
              rgb = this->getPhongColor(rgb, rayStep, normal);
            }

            // front-back compositing
            cOut = cIn + rgb*opacity*(1-opacityIn);
            opacityOut = opacityIn + opacity*(1-opacityIn);

            // Early Ray Termination
            if (opacityOut > 0.9999) {
              // printf("Early Ray Termination at Pixel (%-3zu, %-3zu)", u, v);
              // printf(". RBGA:");
              // printVec(glm::vec4(cOut, opacityOut));
              break;
            }

            cIn = cOut;
            opacityIn = opacityOut;
            ray += rayStep;
          }
          // printf("Pixel (%-3zu, %-3zu): ", u, v);
          // printVec(glm::vec4(cOut, opacityOut));
          this->img.setRGBA(u, v, cOut.r, cOut.g, cOut.b, opacityOut);

          // if (this->img.format == this->img.RGBA) {
          //   this->img.setRGBA(u, v, cOut.r, cOut.g, cOut.b, opacityOut);
          // } else if (this->img.format == this->img.RGB) {
          //   this->img.setRGB(u, v, cOut.r, cOut.g, cOut.b);
          // }
        }
      }
      glm::vec3 ray(rayCorner);
      ray[planeIdx[0]] += (this->img.getWidth()-1)*ustep;
      ray[planeIdx[1]] += (this->img.getHeight()-1)*vstep;
      std::cout << "ray ends at corner";
      printVec(ray);
    }
};

#endif