// #include "container/field.h"
#include "renderer/PieceWiseFunction.h"
#include "renderer/TransferFunction.h"
#include "common.h"
#include <glm/glm.hpp>

#include <string>
#include <vector>
#include <iostream>


// using Eigen::MatrixXd;
int main() {
  // TransferFunction tests
  TransferFunction tf;
  std::cout << "testing RGBA TF:\n";
  tf.addRGBAPoint(0, 0,0,0,0);
  tf.addRGBAPoint(1, 1,1,1,0);
  printVec(tf.getRGBA(-0.1));
  printVec(tf.getRGBA(0));
  printVec(tf.getRGBA(0.2));
  printVec(tf.getRGBA(0.3));
  printVec(tf.getRGBA(1));
  printVec(tf.getRGBA(1.1));
  std::cout << "\n\n";
  std::cout << "testing RGB TF:\n";
  tf.addRGBPoint(0, 0,0,0);
  tf.addRGBPoint(1, 1,1,1);
  printVec(tf.getRGB(-0.1));
  printVec(tf.getRGB(0));
  printVec(tf.getRGB(0.2));
  printVec(tf.getRGB(0.3));
  printVec(tf.getRGB(1));
  printVec(tf.getRGB(1.1));
  std::cout << "\n\n";
  std::cout << "testing Opacity TF:\n";
  tf.addOpacityPoint(0, 0);
  tf.addOpacityPoint(1, 1);
  std::cout << tf.getOpacity(-0.1) << "\n";
  std::cout << tf.getOpacity(0) << "\n";
  std::cout << tf.getOpacity(0.2) << "\n";
  std::cout << tf.getOpacity(0.3) << "\n";
  std::cout << tf.getOpacity(1) << "\n";
  std::cout << tf.getOpacity(1.1) << "\n";


  // printVec(v1*2.f);

  // printVec(lerpGLM(0.2, 0., 1., v1, v2));


  // PieceWiseFunction pf;
  // std::cout << pf.STEP << std::endl;
  // std::cout << pf.LINEAR << std::endl;
  // std::cout <<" inrange?: " <<(int)(pf.isInRange(0.)) << "\n";
  // std::cout <<" inrange?: " <<(int)(pf.isInRange(3.)) << "\n";


  // for (float i = -1; i < 4.; i+=0.179999) {
  // // std::cout << i <<" inrange?: " <<(int)(pf.isInRange(i)) << "\n";

  //   double val = pf.getValue(i);
  //   std::cout << "x=" << i << ", y=" << val  << "\n" << std::endl;
  // }

  // std::cout << pf.getValue(-0.5) << std::endl;
  // std::cout << pf.getValue(-1) << std::endl;
  // std::cout << pf.getValue(-1) << std::endl;
  // std::cout << pf.getValue(-1) << std::endl;
  // std::cout << pf.getValue(-1) << std::endl;

}