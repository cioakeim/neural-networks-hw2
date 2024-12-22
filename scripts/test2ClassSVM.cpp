#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include "osqp/osqp.h"
#include "omp.h"
#include <mkl.h>
#include <Eigen/Dense>
#include "CommonLib/cifarHandlers.hpp"
#include "CommonLib/basicFuncs.hpp"
#include "SVM/SVM.hpp"
#include "SVM/Kernels.hpp"
#include "SVM/Configure.hpp"
#include "CommonLib/eventTimer.hpp"
#include <osqp/osqp.h>


namespace E=Eigen;

static void testeverything(){
  //Load problem data
  OSQPFloat P_x[3] = { 4.0, 1.0, 2.0, };
  OSQPInt   P_nnz  = 3;
  OSQPInt   P_i[3] = { 0, 0, 1, };
  OSQPInt   P_p[3] = { 0, 1, 3, };
  OSQPFloat q[2]   = { 1.0, 1.0, };
  OSQPFloat A_x[4] = { 1.0, 1.0, 1.0, 1.0, };
  OSQPInt   A_nnz  = 4;
  OSQPInt   A_i[4] = { 0, 1, 0, 2, };
  OSQPInt   A_p[3] = { 0, 2, 4, };
  OSQPFloat l[3]   = { 1.0, 0.0, 0.0, };
  OSQPFloat u[3]   = { 1.0, 0.7, 0.7, };
  OSQPInt   n = 2;
  OSQPInt   m = 3;

  //Exitflag
  OSQPInt exitflag;

  //Solver, settings, matrices
  OSQPSolver*   solver   = NULL;
  OSQPSettings* settings = NULL;
  OSQPCscMatrix* P =(OSQPCscMatrix*)malloc(sizeof(OSQPCscMatrix));
  OSQPCscMatrix* A = (OSQPCscMatrix*)malloc(sizeof(OSQPCscMatrix));

  // Populate matrices
  csc_set_data(A, m, n, A_nnz, A_x, A_i, A_p);
  csc_set_data(P, n, n, P_nnz, P_x, P_i, P_p);

  // Set default settings 
  settings = (OSQPSettings *)malloc(sizeof(OSQPSettings));
  if (settings) {
    osqp_set_default_settings(settings);
    settings->polishing = 1;

    //settings->linsys_solver = OSQP_DIRECT_SOLVER;
    //settings->linsys_solver = OSQP_INDIRECT_SOLVER;
  }

  OSQPInt cap = osqp_capabilities();

  printf("This OSQP library supports:\n");
  if(cap & OSQP_CAPABILITY_DIRECT_SOLVER) {
    printf("    A direct linear algebra solver\n");
  }
  if(cap & OSQP_CAPABILITY_INDIRECT_SOLVER) {
    printf("    An indirect linear algebra solver\n");
  }
  if(cap & OSQP_CAPABILITY_CODEGEN) {
    printf("    Code generation\n");
  }
  if(cap & OSQP_CAPABILITY_DERIVATIVES) {
    printf("    Derivatives calculation\n");
  }
  printf("\n");

  // Setup solver
  exitflag = osqp_setup(&solver, P, q, A, l, u, m, n, settings);

  // Solve problem 
  if (!exitflag) exitflag = osqp_solve(solver);

  // Cleanup 
  osqp_cleanup(solver);
  if (A) free(A);
  if (P) free(P);
  if (settings) free(settings);

}

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
