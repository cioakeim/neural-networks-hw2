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
  config.store_path="../data/SVM_models/Multiclass/mock";
  config.training_size=1000;
  config.test_size=1000;
  config.C_list={0.001,0.01,0.1,1,10,100,1000};
  config.kernel_type=RBF;
  config.kernel_parameters.rbf_sigma=1;
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

  // Create log file
  std::ofstream log(config.store_path+"/log.csv");
  if(!log.is_open()){
    std::cerr<<"Error in opening: "<<config.store_path+"/log.csv"<<std::endl;
    exit(1);
  }
  log<<"C,train_accuracy,train_hinge_loss,test_accuracy,test_hinge_loss,SV#"<<"\n";


  for(auto& c: config.C_list){
    EventTimer et;

    multSVM.setCToAll(c);
    float train_hinge_loss,test_hinge_loss;

    et.start("Train multiclass SVM");
    multSVM.trainAllSVMs(train_hinge_loss,test_hinge_loss);
    et.stop();

    std::cout<<"Train hinge loss: "<<train_hinge_loss<<"\nTest Hinge Loss: "<<test_hinge_loss<<std::endl;

    float test_accuracy,train_accuracy;
    et.start("Test on test set");
    multSVM.testOnSet(multSVM.getTestSetRef(), test_accuracy);
    et.stop();
    std::cout<<"Test accuracy: "<<test_accuracy<<std::endl;


    et.start("Test on training set");
    multSVM.testOnSet(multSVM.getTrainingSetRef(), train_accuracy);
    et.stop();
    std::cout<<"Train accuracy: "<<train_accuracy<<std::endl;

    int sv_cnt=multSVM.getTotalSVCount();

    log<<c<<","<<train_accuracy<<","<<train_hinge_loss<<","<<test_accuracy<<","
      <<test_hinge_loss<<","<<sv_cnt<<"\n";

    et.writeToFile(config.store_path+"/C_"+std::to_string(c)+".csv");
  }
  log.close();



  
  script_et.stop();
  script_et.displayIntervals();


  
  return 0;

}
