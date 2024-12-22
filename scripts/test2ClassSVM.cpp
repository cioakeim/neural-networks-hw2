#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include "omp.h"
#include <Eigen/Dense>
#include "CommonLib/cifarHandlers.hpp"
#include "CommonLib/basicFuncs.hpp"
#include "SVM/SVM.hpp"
#include "SVM/Kernels.hpp"
#include "SVM/Configure.hpp"
#include "CommonLib/eventTimer.hpp"


namespace E=Eigen;

int main(int argc,char* argv[]){
  SVM2ClassConfig config;
  config.store_path="../data/SVM_models/linear_model";
  config.dataset_path="../data/cifar-10-batches-bin";
  config.training_size=40000;
  config.test_size=10000;
  config.class1_id=0;
  config.class2_id=2;
  //config.C_list={1e-3,5e-3,1e-2,1e-1,1,5,10,50,100,1000};
  config.C_list={0.1,1,2,5,10,100};
  config.kernel_parameters.poly_c=1;
  config.kernel_parameters.poly_d=2;
  config.kernel_parameters.rbf_sigma=0.1;
  config.kernel_type=LINEAR;
  configureFromArguments(argc,argv,config);
  std::cout<<"Configuration done"<<std::endl;

  EventTimer et;
  et.start("Script total time");

  //mkl_set_num_threads_local(8);
  //std::cout<<"Threads: "<<mkl_get_max_threads()<<std::endl;

  Cifar10Handler c10=Cifar10Handler(config.dataset_path);
  SampleMatrix training_set=c10.getTrainingMatrix(config.training_size);
  SampleMatrix test_set=c10.getTestMatrix(config.test_size);

  normalizeSet(training_set);
  normalizeSet(test_set);
  //return (int)exitflag;


  SampleMatrix train_1v1=extract1v1Dataset(training_set,
                                           config.class1_id,config.class2_id);
  SampleMatrix test_1v1=extract1v1Dataset(test_set,
                                          config.class1_id,config.class2_id);

  /**
  training_set.vectors.resize(0,0);
  training_set.labels.resize(0);
  test_set.vectors.resize(0,0);
  test_set.labels.resize(0);
  */

  SVM svm=SVM(train_1v1,test_1v1);

  switch(config.kernel_type){
  case LINEAR:
    svm.setKernelFunction(linear_kernel,config.kernel_parameters);
    break;
  case POLY:
    svm.setKernelFunction(polynomial_kernel,config.kernel_parameters);
    break;
  case RBF:
    svm.setKernelFunction(rbf_kernel,config.kernel_parameters);
    break;

  }

  svm.setFolderPath(config.store_path);
  ensure_a_path_exists(config.store_path);
  // Store the config info
  storeConfigInfo(config,config.store_path);
  std::ofstream log(config.store_path+"/log.csv");
  if(!log.is_open()){
    std::cerr<<"Error in opening: "<<config.store_path+"/log.csv"<<std::endl;
    exit(1);
  }
  log<<"C,train_accuracy,train_hinge_loss,test_accuracy,test_hinge_loss"<<"\n";

  float best_accuracy=-INFINITY;
  for(auto c: config.C_list){
    // Set C 
    svm.setC(c);
    // Solve and store solution
    svm.solveAndStore();

    if(svm.areSupportVectorsEmpty()){
      std::cout<<"No SVs were found in this run."<<std::endl;
      continue;
    }
    // Test on training set 
    float train_accuracy,train_hinge_loss;
    svm.startEvent("Test on training set");
    svm.testOnSet(svm.getTrainingSetRef(),train_accuracy,train_hinge_loss);
    svm.stopEvent();
    // Test on test set
    float test_accuracy,test_hinge_loss;
    svm.startEvent("Test on test set");
    svm.testOnSet(svm.getTestSetRef(),test_accuracy,test_hinge_loss);
    svm.stopEvent(); 

    // Log results
    log<<c<<","<<train_accuracy<<","<<train_hinge_loss<<","
      <<test_accuracy<<","<<test_hinge_loss<<"\n";

    // If it's the best so far, store
    if(test_accuracy>best_accuracy){
      svm.storeToFile();
    }
    svm.clearSolution();

    svm.displayCurrentIntervals();
    svm.storeEventsToFile();
  }
  log.close();
  et.stop();
  et.displayIntervals();
  return 0;

}
