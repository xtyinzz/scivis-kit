#ifndef VOLUMERENDERER_H
#define VOLUMERENDERER_H

#include "VolumeRenderer.h"
#include "TransferFunction.h"
#include <CImg.h>
// this used to prevent redefinition of Success
#ifdef Success
  #undef Success
#endif

using namespace cimg_library; 

class VolumeRenderer {
  private:
    
  public:
    int tmp;
    CImg<unsigned char> image;
    TransferFunction tf;
    
    VolumeRenderer();

    void setImageDimension(int row, int col);
    void render();
};

#endif