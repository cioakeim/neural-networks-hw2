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
#include "CommonLib/basicFuncs.hpp"
#include "SVM/MultiClassSVM.hpp"

namespace E=Eigen;

int main(){
  std::string cifar_path="../data/cifar-10-batches-bin";
  std::string model_path="../data/SVM_models";
  int training_size=5000;
  int test_size=100;

  EventTimer et;
  et.start("Total script time");


  std::cout<<"Hi"<<std::endl;


  Cifar10Handler c10=Cifar10Handler(cifar_path);
  SampleMatrix training_set=c10.getTrainingMatrix(training_size);
  SampleMatrix test_set=c10.getTestMatrix(test_size);

  normalizeSet(training_set);
  normalizeSet(test_set);



  MultiClassSVM multSVM(training_set,test_set);
  multSVM.setNameAndPath("test_name_store", model_path);

  float C=1000;
  multSVM.setCToAll(C);
  KernelParameters kernel_parameters;
  kernel_parameters.rbf_gamma=1;
  multSVM.setKernelToAll(rbf_kernel,kernel_parameters);

  /**
  float accuracy,mean_hinge_loss;
  multSVM.trainTwoClassSVM(0,1);
  std::cout<<"Training successful"<<std::endl;
  multSVM.testTwoClassSVM(0, 1, (multSVM.getSVMPointer(0,1))->getTrainingSetRef()
                          ,accuracy, mean_hinge_loss);
  */

  multSVM.trainAllSVMs();


  
  et.stop();
  et.displayIntervals();


  
  return 0;

}
