#include "SVM/Kernels.hpp"
#include <iostream>


E::MatrixXf linear_kernel(const E::MatrixXf& a,const E::MatrixXf& b,
                          KernelParameters param){
  return a.transpose()*b;
}


E::MatrixXf polynomial_kernel(const E::MatrixXf& a,const E::MatrixXf& b,
                              KernelParameters param){
  const E::MatrixXf temp=(a.transpose()*b).array()/(a.rows());
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
  const float feature_size=a.rows();
  const float sigma=param.rbf_sigma*std::sqrt(feature_size);
  const float gamma=0.5*(1/(sigma*sigma));


  #pragma omp parallel for
  for(int i=0;i<size_1;i++){
    #pragma omp parallel for
    for(int j=0;j<size_2;j++){
      float dist_sq=(a.col(i)-b.col(j)).squaredNorm();
      result(i,j)=std::exp(-(gamma*dist_sq));
    }
  }
  if(result.array().isNaN().sum()>0){
    std::cerr<<"NaNs found in kernel calculation!!"<<std::endl;
    exit(1);
  }
  return result;
}

