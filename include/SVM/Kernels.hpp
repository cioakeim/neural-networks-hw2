#ifndef KERNELS_HPP
#define KERNELS_HPP

#ifndef EIGEN_USE_BLAS
#define EIGEN_USE_BLAS
#endif
#ifndef EIGEN_USE_LAPACK
#define EIGEN_USE_LAPACK
#endif
#include <Eigen/Dense>

namespace E=Eigen;

struct KernelParameters{
  int poly_d;
  float poly_c;

  float rbf_sigma;
};

// In all implementations a is input and b is SV set

E::MatrixXf linear_kernel(const E::MatrixXf& a,const E::MatrixXf& b,
                          KernelParameters param);

E::MatrixXf polynomial_kernel(const E::MatrixXf& a,const E::MatrixXf& b,
                              KernelParameters param);

E::MatrixXf rbf_kernel(const E::MatrixXf& a, const E:: MatrixXf& b,
                       KernelParameters param);


#endif
