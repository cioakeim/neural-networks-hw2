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
#include "SVM/Configure.hpp"

namespace E=Eigen;

int main(int argc,char* argv[]){
  SVM2ClassConfig config;
  config.training_type=MultiClass;
  config.dataset_path="../data/cifar-10-batches-bin";
  config.store_path="../data/SVM_models/big_test";
  config.training_size=40000;
  config.test_size=8000;
  config.C_list={0.001,0.01,0.1,1,10,100,1000};
  config.C_list={5};
  config.kernel_type=RBF;
  config.kernel_parameters.rbf_sigma=0.2;
  configureFromArguments(argc, argv, config);

  EventTimer script_et;
  script_et.start("Total script time");


  std::cout<<"Hi"<<std::endl;


  Cifar10Handler c10=Cifar10Handler(config.dataset_path);
  SampleMatrix training_set=c10.getTrainingMatrix(config.training_size);
  SampleMatrix test_set=c10.getTestMatrix(config.test_size);

  normalizeSet(training_set);
  normalizeSet(test_set);


  MultiClassSVM multSVM(training_set,test_set);
  multSVM.setPath(config.store_path);
  storeConfigInfo(config,config.store_path);

  switch(config.kernel_type){
  case RBF:
    multSVM.setKernelToAll(rbf_kernel,config.kernel_parameters);
    break;
  case POLY:
    multSVM.setKernelToAll(polynomial_kernel,config.kernel_parameters);
    break;
  case LINEAR:
    multSVM.setKernelToAll(linear_kernel,config.kernel_parameters);
    break;
  }

  EventTimer et;

  std::cout<<"Train"<<std::endl;
  et.start("Train multi-class");
  multSVM.trainAllSVMs(config.C_list);
  et.stop();
  std::cout<<"Done."<<std::endl;

  // Create log file
  std::ofstream log(config.store_path+"/log.csv");
  if(!log.is_open()){
    std::cerr<<"Error in opening: "<<config.store_path+"/log.csv"<<std::endl;
    exit(1);
  }
  log<<"C,train_accuracy,train_hinge_loss,test_accuracy,test_hinge_loss,SV#,SVmean,SVstd"<<"\n";

  for(auto c: config.C_list){
    std::cout<<"Load"<<std::endl;
    et.start("Loading");
    multSVM.loadSVMs(c);
    et.stop();
    std::cout<<"Done"<<std::endl;

    float train_accuracy,train_hinge_loss;
    train_accuracy=train_hinge_loss=-1;
    /*
    std::cout<<"Test1"<<std::endl;
    et.start("Test on training set");
    multSVM.testOnSet(training_set,train_accuracy,train_hinge_loss);
    et.stop();
    std::cout<<"Done"<<std::endl;
    */

    std::cout<<"Res: "<<train_accuracy<<" "<<train_hinge_loss<<std::endl;

    std::cout<<"Test2"<<std::endl;
    et.start("Test on test set");
    float test_accuracy,test_hinge_loss;
    multSVM.testOnSet(test_set,test_accuracy,test_hinge_loss);
    et.stop();
    std::cout<<"Done"<<std::endl;
    std::cout<<"Res: "<<test_accuracy<<" "<<test_hinge_loss<<std::endl;

    int sum;
    float mean,sigma;
    multSVM.getTotalSVStats(sum,mean,sigma);

    log<<c<<","<<train_accuracy<<","<<train_hinge_loss<<","
      <<test_accuracy<<","<<test_hinge_loss<<","
      <<sum<<","<<mean<<","<<sigma<<"\n";
  }

  log.close();
  et.displayIntervals();

  
  script_et.stop();
  script_et.displayIntervals();


  
  return 0;

}
