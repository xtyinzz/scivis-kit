#ifndef SVK_COMMON_NUMERICAL
#define SVK_COMMON_NUMERICAL

// #include "container/field.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>
#include <type_traits>
#define EPS 1e-8

using namespace Eigen;

////////////////////////////////// Interpolation ///////////////////////////////


//		  ________
//		0        1
template <typename T>
inline T lerp(T v0, T v1, T alpha) {
  T p = (1-alpha)*v0 + v1*alpha;
  return p;
}

//		  ________
//		0        1

template <typename T>
inline T lerpGLM(float x, float x0, float x1, T v0, T v1) {
  float alpha = (x-x0) / (x1 - x0);
  T p = (1-alpha)*v0 + v1*alpha;
  return p;
}

template <typename T>
inline T lerp(float x, float x0, float x1, T v0, T v1) {
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
inline T bilerp(T v0, T v1, T v2, T v3, T alpha, T beta) {
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
inline T trilerp(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7,
          float alpha, float beta, float gamma) {
  T p = (1-alpha)*(1-beta)*(1-gamma)*v0 + 
        alpha*(1-beta)*(1-gamma)*v1 + 
        (1-alpha)*beta*(1-gamma)*v2 + 
        alpha*beta*(1-gamma)*v3 + 
        (1-alpha)*(1-beta)*gamma*v4 + 
        alpha*(1-beta)*gamma*v5 + 
        (1-alpha)*beta*gamma*v6 +
        alpha*beta*gamma*v7;
  // std::cout << "TRILERP:\n" << v0 << "\n**\n**\n" << v1<<"\n**\n**\n" <<v2<<"\n**\n**\n" <<v3<<"\n**\n**\n" <<v4<<"\n**\n**\n" <<v5<<"\n**\n**\n" <<v6<<"\n**\n**\n" <<v7<< "\nRESUlt\n" << p << "\n";
  return p;
}

// Alternative algorithm in https://en.wikipedia.org/wiki/Trilinear_interpolation
template <typename T, typename REAL>
std::vector<std::vector<T>> trilerpSysEqCoeff(const Array3<REAL> &coord, const Array3<REAL> &lowVtx, const Array3<REAL> &highVtx, std::vector<T> cell_val) {
  Array3<REAL> tmpcoord = (coord - lowVtx) / (highVtx - lowVtx);
  // this cumbersome implement is to wait for figuring out perform trilper w/o normalizing the coordiantes.
  Array3<REAL> tmplowVtx = {0., 0., 0.};
  Array3<REAL> tpmphighVtx = {1., 1., 1.};
  float x = tmpcoord[0];
  float y = tmpcoord[1];
  float z = tmpcoord[2];

  float x0 = tmplowVtx[0];
  float y0 = tmplowVtx[1];
  float z0 = tmplowVtx[2];

  float x1 = tpmphighVtx[0];
  float y1 = tpmphighVtx[1];
  float z1 = tpmphighVtx[2];
  T v000 = cell_val[0];
  T v100 = cell_val[1];
  T v010 = cell_val[2];
  T v110 = cell_val[3];
  T v001 = cell_val[4];
  T v101 = cell_val[5];
  T v011 = cell_val[6];
  T v111 = cell_val[7];
  // T a0 = 
  //   (-v000*x1*y1*z1 + v001*x1*y1*z0 + v010*x1*y0*z1 - v011*x1*y0*z0 + 
  //   v100*x0*y1*z1 - v101*x0*y1*z0 - v110*x0*y0*z1 + v111*x0*y0*z0);

  // T a1 = 
  //   (v000*z1*y1 - v001*z0*y1 - v010*z1*y0 + v011*z0*y0 - 
  //   v100*z1*y1 + v101*z0*y1 + v110*z1*y0 - v111*z0*y0);

  // T a2 = 
  //   (v000*x1*z1 - v001*x1*z0 - v010*x1*z1 + v011*x1*z0 - 
  //   v100*x0*z1 + v101*x0*z0 + v110*x0*z1 - v111*x0*z0);

  // T a3 = 
  //   (v000*x1*y1 - v001*x1*y1 - v010*x1*y0 + v011*x1*y0 - 
  //   v100*x0*y1 + v101*x0*y1 + v110*x0*y0 - v111*x0*y0);

  // T a4 = 
  //   (-v000*z1 + v001*z0 + v010*z1 - v011*z0 + 
  //   v100*z1 - v101*z0 - v110*z1 + v111*z0);

  // T a5 = 
  //   (-v000*y1 + v001*y1 + v010*y0 - v011*y0 + 
  //   v100*y1 - v101*y1 - v110*y0 + v111*y0);

  // T a6 = 
  //   (-v000*x1 + v001*x1 + v010*x1 - v011*x1 + 
  //   v100*x0 - v101*x0 - v110*x0 + v111*x0);

  // T a7 = 
  //   (v000 - v001 - v010 + v011 -v100 + v101 + v110 - v111);
  T a0 = v000;

  T a1 = -v000 + v100;

  T a2 = -v000 + v010;

  T a3 = -v000 + v001;

  T a4 = v000 - v100 + v110 - v010;

  T a5 = v000 - v100 - v001 + v101;

  T a6 = v000 - v010 - v001 + v011;

  T a7 = -v000 + v100 - v110 + v010 + v001 - v101 + v111 - v011;

  T interpolant = (a0 + a1*x + a2*y + a3*z +
          a4*x*y + a5*x*z + a6*y*z + a7*x*y*z);

  return std::vector<std::vector<T>>{ {interpolant}, {a0, a1, a2, a3, a4, a5, a6, a7} };
}

template <typename T, typename REAL>
inline T trilerpSysEq(REAL x, REAL y, REAL z, std::vector<T> coeff) {
  return (coeff[0] + coeff[1]*x + coeff[2]*y + coeff[3]*z +
          coeff[4]*x*y + coeff[5]*x*z + coeff[6]*y*z + coeff[7]*x*y*z);
}


// // Alternative algorithm in https://en.wikipedia.org/wiki/Trilinear_interpolation
// template <typename T>
// std::vector<T> trilerpSysEqCoeff(Array3d coord, Array3d lowVtx, Array3d highVtx, std::vector<T> cell_val) {
//   coord = (coord - lowVtx) / (highVtx - lowVtx);
//   // this cumbersome implement is to wait for figuring out perform trilper w/o normalizing the coordiantes.
//   lowVtx = {0., 0., 0.};
//   highVtx = {1., 1., 1.};
//   double x = coord[0];
//   double y = coord[1];
//   double z = coord[2];
//   double x0 = lowVtx[0];
//   double y0 = lowVtx[1];
//   double z0 = lowVtx[2];
//   double x1 = highVtx[0];
//   double y1 = highVtx[1];
//   double z1 = highVtx[2];
//   T v000 = cell_val[0];
//   T v100 = cell_val[1];
//   T v010 = cell_val[2];
//   T v110 = cell_val[3];
//   T v001 = cell_val[4];
//   T v101 = cell_val[5];
//   T v011 = cell_val[6];
//   T v111 = cell_val[7];
//   T a0 = 
//     (-v000*x1*y1*z1 + v001*x1*y1*z0 + v010*x1*y0*z1 - v011*x1*y0*z0 + 
//     v100*x0*y1*z1 - v101*x0*y1*z0 - v110*x0*y0*z1 + v111*x0*y0*z0) / ((x0-x1)*(y0-y1)*(z0-z1));

//   T a1 = 
//     (v000*z1*y1 - v001*z0*y1 - v010*z1*y0 + v011*z0*y0 - 
//     v100*z1*y1 + v101*z0*y1 + v110*z1*y0 - v111*z0*y0) / ((x0-x1)*(y0-y1)*(z0-z1));

//   T a2 = 
//     (v000*x1*z1 - v001*x1*z0 - v010*x1*z1 + v011*x1*z0 - 
//     v100*x0*z1 + v101*x0*z0 + v110*x0*z1 - v111*x0*z0) / ((x0-x1)*(y0-y1)*(z0-z1));

//   T a3 = 
//     (v000*x1*y1 - v001*x1*y1 - v010*x1*y0 + v011*x1*y0 - 
//     v100*x0*y1 + v101*x0*y1 + v110*x0*y0 - v111*x0*y0) / ((x0-x1)*(y0-y1)*(z0-z1));

//   T a4 = 
//     (-v000*z1 + v001*z0 + v010*z1 - v011*z0 + 
//     v100*z1 - v101*z0 - v110*z1 + v111*z0) / ((x0-x1)*(y0-y1)*(z0-z1));

//   T a5 = 
//     (-v000*y1 + v001*y1 + v010*y0 - v011*y0 + 
//     v100*y1 - v101*y1 - v110*y0 + v111*y0) / ((x0-x1)*(y0-y1)*(z0-z1));

//   T a6 = 
//     (-v000*x1 + v001*x1 + v010*x1 - v011*x1 + 
//     v100*x0 - v101*x0 - v110*x0 + v111*x0) / ((x0-x1)*(y0-y1)*(z0-z1));

//   T a7 = 
//     (v000 - v001 - v010 + v011 -v100 + v101 + v110 - v111) / ((x0-x1)*(y0-y1)*(z0-z1));

//   return std::vector<T>{a0, a1, a2, a3, a4, a5, a6, a7};
// }

// template <typename T>
// inline T trilerpSysEq(double x, double y, double z, std::vector<T> coeff) {
//   return (coeff[0] + coeff[1]*x + coeff[2]*y + coeff[3]*z +
//           coeff[4]*x*y + coeff[5]*x*z + coeff[6]*y*z + coeff[7]*x*y*z);
// }


////////////////////////////////// Precision & Error Tolerance ///////////////////////////////


inline bool isClose(float a, float b, double atol=1.48e-8, double rtol=0.0) {
  return abs(a-b) < (atol + rtol*b);
}
inline bool isClose(double a, double b, double atol=1.48e-8, double rtol=0.0) {
  return abs(a-b) < (atol + rtol*b);
}

template <typename T>
inline ArrayX<bool> isClose(const Array3<T> &a, const Array3<T> &b, double atol=1.48e-8, double rtol=0.0) {
  // std::cout <<"a=" << a << " b=" << b << "--" << abs(a-b) << "heihei  " << (atol + rtol*b) << "haha \n";
  // std::cout << "atol " << atol << "\n";
  // std::cout << "rtol " << rtol << "\n";
  // std::cout << "rtol*b  " << rtol*b << "\n";
  return abs(a-b) < (atol + rtol*b);
}

template <typename T>
inline bool allClose(const Array3<T> &a, const Array3<T> &b, double atol=1.48e-8, double rtol=0.0) {
  return isClose(a, b, atol, rtol).isOnes(0);
}

template <typename T>
inline T closeRound(T a, T b, double atol=1.48e-8, double rtol=0.0) {
  T result = a;
  if (isClose(a, b, atol, rtol)) {
    result = b;
  }
  return result;
}

template <typename T>
inline T trigApprox(T (*trigFunc)(T), T val, double atol=1.48e-8, double rtol=0.0) {
  T result = trigFunc(val);
  result = closeRound(result, (T)0., atol, rtol);
  result = closeRound(result, (T)1., atol, rtol);
  result = closeRound(result, (T)-1., atol, rtol);
  return result;
}

template <typename T>
inline T radians(T deg) {
  return deg * M_PI / 180.;
}

template <typename T>
inline Array3<T> sph2car(T r, T theta, T phi) {
  Array3<T> sph{
    r*trigApprox(std::sin, phi)*trigApprox(std::cos, theta),
    r*trigApprox(std::sin, phi)*trigApprox(std::sin, theta),
    r*trigApprox(std::cos, phi)
  };
  return sph;
}

////////////////////////////////// Particle Tracing ///////////////////////////////


// forward declare VectorField for particle tracing methods
template <typename T>
class VectorField;

template <typename T, typename VecT>
std::vector<Array3<T>> particleTracingRK1(Array3<T> seed, VectorField<VecT> *vecField, double stepsize, int maxstep) {
  std::vector<Array3<T>> traces;
  traces.push_back(seed);
  for (int i = 0; i < maxstep; i++) {
    seed += stepsize*(vecField->getVal(seed[0], seed[1], seed[2]).array());
    // exit if out of bound
    if (!vecField->isBounded(seed[0], seed[1], seed[2]))
      break;
    traces.push_back(seed);
  }
  return traces;
}

template <typename T, typename VecT>
std::vector<Array3<T>> particleTracingRK4(Array3<T> seed, VectorField<VecT> *vecField, double stepsize, int maxstep) {
  std::vector<Array3<T>> traces;
  traces.push_back(seed);
  for (int i = 0; i < maxstep; i++) {
    Array3<T> k1 = vecField->getVal(seed[0], seed[1], seed[2]).array();
    // if (k1 == NULL) break;
    Array3<T> k1_pos = seed + 0.5*stepsize*k1;

    Array3<T> k2 = vecField->getVal(k1_pos[0], k1_pos[1], k1_pos[2]).array();
    // if (k2 == NULL) break;
    Array3<T> k2_pos = seed + 0.5*stepsize*k2;

    Array3<T> k3 = vecField->getVal(k2_pos[0], k2_pos[1], k2_pos[2]).array();
    // if (k3 == NULL) break;
    Array3<T> k3_pos = seed + stepsize*k3;

    Array3<T> k4 = vecField->getVal(k3_pos[0], k3_pos[1], k3_pos[2]).array();
    // if (k4 == NULL) break;

    seed = seed + stepsize*(k1 + 2*k2 + 2*k3 + k4)/6.;
    // exit if out of bound
    if (!vecField->isBounded(seed[0], seed[1], seed[2]))
      break;
    traces.push_back(seed);
  }
  return traces;
}

template <typename T>
Vector3<T> calcVort(const Matrix3<T> &jacobian) {
  Vector3f vorticity = {
    jacobian(2, 1) - jacobian(1, 2),
    jacobian(0, 2) - jacobian(2, 0),
    jacobian(1, 0) - jacobian(0, 1)
  };
  return vorticity;
}

#endif