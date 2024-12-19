#include <iostream>
#include <vector>
#include <string>
#include "osqp/osqp.h"
#include "omp.h"
#include <Eigen/Dense>
#include "CommonLib/cifarHandlers.hpp"
#include "SVM/SVM.hpp"
#include "SVM/Kernels.hpp"
#include "CommonLib/eventTimer.hpp"

namespace E=Eigen;

std::vector<int> findIndices(E::VectorXi& labels, int key1, int key2){
  std::vector<int> result;
  for(int i=0;i<labels.size();i++){
    if(labels(i)==key1 || labels(i)==key2){
      result.push_back(i);
    }
  }
  return result;
}

int main(){
  std::string cifar_path="../data/cifar-10-batches-bin";
  int training_size=5000;
  int test_size=100;

  //omp_set_num_threads(2);

  EventTimer et;
  et.start("Script total time");

  std::cout<<"Hi"<<std::endl;


  Cifar10Handler c10=Cifar10Handler(cifar_path);
  SampleMatrix training_set=c10.getTrainingMatrix(training_size);
  SampleMatrix test_set=c10.getTestMatrix(test_size);

  float sigma=std::sqrt(training_set.vectors.array().pow(2).mean());
  float mean=training_set.vectors.array().mean();
  training_set.vectors.array()-=mean;
  training_set.vectors.array()/=sigma;

  int label1=0,label2=1;
  std::vector<int> idx1=findIndices(training_set.labels,label1,label2);
  std::cout<<"Sizes: "<<idx1.size()<<std::endl;
  SampleMatrix two_class_set;
  two_class_set.vectors=E::MatrixXf(training_set.vectors.rows(),idx1.size());
  two_class_set.labels=E::VectorXi(idx1.size());
  const int size=idx1.size();
  #pragma omp parallel for
  for(int i=0;i<size;i++){
    two_class_set.vectors.col(i)=training_set.vectors.col(idx1[i]);
    two_class_set.labels(i)=(training_set.labels(idx1[i])==label1)?+1:-1;
  }



  SVM svm=SVM(two_class_set,test_set);

  // Kernel config
  KernelParameters kernel_parameters;
  kernel_parameters.poly_c=1;
  kernel_parameters.poly_d=2;
  kernel_parameters.rbf_gamma=0.1;
  //svm.setKernelFunction(linear_kernel,kernel_parameters);
  //svm.setKernelFunction(polynomial_kernel,kernel_parameters);
  svm.setKernelFunction(rbf_kernel,kernel_parameters);

  std::vector<float> C_list={10};

  for(auto c: C_list){
    svm.setC(c);

    svm.solveAndStore();
    std::cout<<"This is good"<<std::endl;

    float accuracy,mean_hinge_loss;
    svm.testOnSet(svm.getTrainingSetRef(), accuracy, mean_hinge_loss);

    svm.displayCurrentIntervals();
  }

  et.stop();
  et.displayIntervals();

  
  return 0;

}
