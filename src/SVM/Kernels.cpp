#include "SVM/Kernels.hpp"
#include <iostream>


E::MatrixXf linear_kernel(const E::MatrixXf& a,const E::MatrixXf& b,
                          KernelParameters param){
  return a.transpose()*b;
}


E::MatrixXf polynomial_kernel(const E::MatrixXf& a,const E::MatrixXf& b,
                              KernelParameters param){
  const E::MatrixXf temp=a.transpose()*b;
  return (temp.array()+param.poly_c).array().pow(param.poly_d);
}

E::MatrixXf rbf_kernel(const E::MatrixXf& a, const E:: MatrixXf& b,
                       KernelParameters param){
  E::MatrixXf result(a.cols(),b.cols());
  const int size_1=a.cols();
  const int size_2=b.cols();
  /**
  for(int i=0;i<size;i++){
    result.row(i)=(((b.colwise()-a.col(i)).colwise().norm()).array()*param.rbf_gamma).exp();
    std::cout<<"size "<<result.row(i).size()<<std::endl;
  }
  **/
  #pragma omp parallel for
  for(int i=0;i<size_1;i++){
    #pragma omp parallel for
    for(int j=0;j<size_2;j++){
      float dist_sq=(a.col(i)-b.col(j)).squaredNorm();
      result(i,j)=std::exp(-param.rbf_gamma*dist_sq);
    }
  }
  return result;
}

