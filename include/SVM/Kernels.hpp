#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <Eigen/Dense>

namespace E=Eigen;

struct KernelParameters{
  int poly_d;
  float poly_c;

  float rbf_gamma;
};

// In all implementations a is input and b is SV set

E::MatrixXf linear_kernel(const E::MatrixXf& a,const E::MatrixXf& b,
                          KernelParameters param);

E::MatrixXf polynomial_kernel(const E::MatrixXf& a,const E::MatrixXf& b,
                              KernelParameters param);

E::MatrixXf rbf_kernel(const E::MatrixXf& a, const E:: MatrixXf& b,
                       KernelParameters param);


#endif
