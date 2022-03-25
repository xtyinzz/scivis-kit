#ifndef SVK_COMMON_NUMERICAL
#define SVK_COMMON_NUMERICAL

#include <Eigen/Dense>
#include <vector>
#define EPS 1e-8
using namespace Eigen;

//		  ________
//		0        1
template <typename T>
T lerp(T v0, T v1, T alpha) {
  T p = (1-alpha)*v0 + v1*alpha;
  return p;
}

//		  ________
//		0        1

template <typename T>
T lerpGLM(float x, float x0, float x1, T v0, T v1) {
  float alpha = (x-x0) / (x1 - x0);
  T p = (1-alpha)*v0 + v1*alpha;
  return p;
}

template <typename T>
T lerp(float x, float x0, float x1, T v0, T v1) {
  float alpha = (x-x0) / (x1 - x0);
  T p = (1-alpha)*v0 + v1*alpha;
  return p;
}

//		  2  ________ 3
//		   /        /
//		  /        /
//		 /________/
//		0        1
template <typename T>
T bilerp(T v0, T v1, T v2, T v3, T alpha, T beta) {
  T p = (1-alpha)*(1-beta)*v0 + 
        alpha*(1-beta)*v1 + 
        (1-alpha)*beta*v2 + 
        alpha*beta*v3;
  return p;
}

  //
  //		    6________7
  //		   /|       /|
  //		  / |      / |
  //		4/_______5/  |
  //		|  2|___ |___|3
  //		|  /     |  /
  //		| /      | /
  //		|/_______|/
  //		0        1
template <typename T>
T trilerp(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7,
          float alpha, float beta, float gamma) {
  T p = (1-alpha)*(1-beta)*(1-gamma)*v0 + 
        alpha*(1-beta)*(1-gamma)*v1 + 
        (1-alpha)*beta*(1-gamma)*v2 + 
        alpha*beta*(1-gamma)*v3 + 
        (1-alpha)*(1-beta)*gamma*v4 + 
        alpha*(1-beta)*gamma*v5 + 
        (1-alpha)*beta*gamma*v6 +
        alpha*beta*gamma*v7;
  return p;
}

// Alternative algorithm in https://en.wikipedia.org/wiki/Trilinear_interpolation
template <typename T>
std::vector<T> trilerpSysEqCoeff(Array3f coord, Array3f lowVtx, Array3f highVtx, std::vector<T> cell_val) {
  coord = (coord - lowVtx) / (highVtx - lowVtx);
  // this cumbersome implement is to wait for figuring out perform trilper w/o normalizing the coordiantes.
  lowVtx = {0., 0., 0.};
  highVtx = {1., 1., 1.};
  float x = coord[0];
  float y = coord[1];
  float z = coord[2];
  float x0 = lowVtx[0];
  float y0 = lowVtx[1];
  float z0 = lowVtx[2];
  float x1 = highVtx[0];
  float y1 = highVtx[1];
  float z1 = highVtx[2];
  T v000 = cell_val[0];
  T v100 = cell_val[1];
  T v010 = cell_val[2];
  T v110 = cell_val[3];
  T v001 = cell_val[4];
  T v101 = cell_val[5];
  T v011 = cell_val[6];
  T v111 = cell_val[7];
  T a0 = 
    (-v000*x1*y1*z1 + v001*x1*y1*z0 + v010*x1*y0*z1 - v011*x1*y0*z0 + 
    v100*x0*y1*z1 - v101*x0*y1*z0 - v110*x0*y0*z1 + v111*x0*y0*z0) / ((x0-x1)*(y0-y1)*(z0-z1));

  T a1 = 
    (v000*z1*y1 - v001*z0*y1 - v010*z1*y0 + v011*z0*y0 - 
    v100*z1*y1 + v101*z0*y1 + v110*z1*y0 - v111*z0*y0) / ((x0-x1)*(y0-y1)*(z0-z1));

  T a2 = 
    (v000*x1*z1 - v001*x1*z0 - v010*x1*z1 + v011*x1*z0 - 
    v100*x0*z1 + v101*x0*z0 + v110*x0*z1 - v111*x0*z0) / ((x0-x1)*(y0-y1)*(z0-z1));

  T a3 = 
    (v000*x1*y1 - v001*x1*y1 - v010*x1*y0 + v011*x1*y0 - 
    v100*x0*y1 + v101*x0*y1 + v110*x0*y0 - v111*x0*y0) / ((x0-x1)*(y0-y1)*(z0-z1));

  T a4 = 
    (-v000*z1 + v001*z0 + v010*z1 - v011*z0 + 
    v100*z1 - v101*z0 - v110*z1 + v111*z0) / ((x0-x1)*(y0-y1)*(z0-z1));

  T a5 = 
    (-v000*y1 + v001*y1 + v010*y0 - v011*y0 + 
    v100*y1 - v101*y1 - v110*y0 + v111*y0) / ((x0-x1)*(y0-y1)*(z0-z1));

  T a6 = 
    (-v000*x1 + v001*x1 + v010*x1 - v011*x1 + 
    v100*x0 - v101*x0 - v110*x0 + v111*x0) / ((x0-x1)*(y0-y1)*(z0-z1));

  T a7 = 
    (v000 - v001 - v010 + v011 -v100 + v101 + v110 - v111) / ((x0-x1)*(y0-y1)*(z0-z1));

  return std::vector<T>{a0, a1, a2, a3, a4, a5, a6, a7};
}

template <typename T>
inline T trilerpSysEq(float x, float y, float z, std::vector<T> coeff) {
  return (coeff[0] + coeff[1]*x + coeff[2]*y + coeff[3]*z +
          coeff[4]*x*y + coeff[5]*x*z + coeff[6]*y*z + coeff[7]*x*y*z);
}

inline ArrayX<bool> isClose(ArrayXf a, ArrayXf b, float atol=1.48-8, float rtol=0.0) {
  return a.absolute_difference(b) < (atol + rtol*b);
}

inline bool allClose(ArrayXf a, ArrayXf b, float atol=1.48-8, float rtol=0.0) {
  return isClose(a, b, atol, rtol).isOnes(0);
}

// Alternative algorithm in https://en.wikipedia.org/wiki/Trilinear_interpolation
template <typename T>
std::vector<T> trilerpSysEqCoeff(Array3d coord, Array3d lowVtx, Array3d highVtx, std::vector<T> cell_val) {
  coord = (coord - lowVtx) / (highVtx - lowVtx);
  // this cumbersome implement is to wait for figuring out perform trilper w/o normalizing the coordiantes.
  lowVtx = {0., 0., 0.};
  highVtx = {1., 1., 1.};
  double x = coord[0];
  double y = coord[1];
  double z = coord[2];
  double x0 = lowVtx[0];
  double y0 = lowVtx[1];
  double z0 = lowVtx[2];
  double x1 = highVtx[0];
  double y1 = highVtx[1];
  double z1 = highVtx[2];
  T v000 = cell_val[0];
  T v100 = cell_val[1];
  T v010 = cell_val[2];
  T v110 = cell_val[3];
  T v001 = cell_val[4];
  T v101 = cell_val[5];
  T v011 = cell_val[6];
  T v111 = cell_val[7];
  T a0 = 
    (-v000*x1*y1*z1 + v001*x1*y1*z0 + v010*x1*y0*z1 - v011*x1*y0*z0 + 
    v100*x0*y1*z1 - v101*x0*y1*z0 - v110*x0*y0*z1 + v111*x0*y0*z0) / ((x0-x1)*(y0-y1)*(z0-z1));

  T a1 = 
    (v000*z1*y1 - v001*z0*y1 - v010*z1*y0 + v011*z0*y0 - 
    v100*z1*y1 + v101*z0*y1 + v110*z1*y0 - v111*z0*y0) / ((x0-x1)*(y0-y1)*(z0-z1));

  T a2 = 
    (v000*x1*z1 - v001*x1*z0 - v010*x1*z1 + v011*x1*z0 - 
    v100*x0*z1 + v101*x0*z0 + v110*x0*z1 - v111*x0*z0) / ((x0-x1)*(y0-y1)*(z0-z1));

  T a3 = 
    (v000*x1*y1 - v001*x1*y1 - v010*x1*y0 + v011*x1*y0 - 
    v100*x0*y1 + v101*x0*y1 + v110*x0*y0 - v111*x0*y0) / ((x0-x1)*(y0-y1)*(z0-z1));

  T a4 = 
    (-v000*z1 + v001*z0 + v010*z1 - v011*z0 + 
    v100*z1 - v101*z0 - v110*z1 + v111*z0) / ((x0-x1)*(y0-y1)*(z0-z1));

  T a5 = 
    (-v000*y1 + v001*y1 + v010*y0 - v011*y0 + 
    v100*y1 - v101*y1 - v110*y0 + v111*y0) / ((x0-x1)*(y0-y1)*(z0-z1));

  T a6 = 
    (-v000*x1 + v001*x1 + v010*x1 - v011*x1 + 
    v100*x0 - v101*x0 - v110*x0 + v111*x0) / ((x0-x1)*(y0-y1)*(z0-z1));

  T a7 = 
    (v000 - v001 - v010 + v011 -v100 + v101 + v110 - v111) / ((x0-x1)*(y0-y1)*(z0-z1));

  return std::vector<T>{a0, a1, a2, a3, a4, a5, a6, a7};
}

template <typename T>
inline T trilerpSysEq(double x, double y, double z, std::vector<T> coeff) {
  return (coeff[0] + coeff[1]*x + coeff[2]*y + coeff[3]*z +
          coeff[4]*x*y + coeff[5]*x*z + coeff[6]*y*z + coeff[7]*x*y*z);
}

inline ArrayX<bool> isClose(ArrayXd a, ArrayXd b, double atol=1.48-8, double rtol=0.0) {
  return a.absolute_difference(b) < (atol + rtol*b);
}

inline bool allClose(ArrayXd a, ArrayXd b, double atol=1.48-8, double rtol=0.0) {
  return isClose(a, b, atol, rtol).isOnes(0);
}

#endif