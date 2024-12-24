#include <iostream>
#include <vector>
#include <string>
#include "omp.h"
#include <Eigen/Dense>
#include "CommonLib/cifarHandlers.hpp"
#include "CommonLib/eventTimer.hpp"
#include "CommonLib/basicFuncs.hpp"
#include "MLP/MultiBinaryMLP.hpp"

namespace E=Eigen;

int main(int argc,char* argv[]){
  std::string dataset_path="../data/cifar-10-batches-bin";
  std::string store_path="../data/MLPS/MBMLP";
  int training_size=50000;
  int test_size=10000;
  float learning_rate=0.01;
  int batch_size=50;
  int epoch_number=100;
  int epoch_step=5;


  EventTimer script_et;
  script_et.start("Total script time");



  std::cout<<"HI"<<std::endl;

  Cifar10Handler c10=Cifar10Handler(dataset_path);
  SampleMatrix training_set=c10.getTrainingMatrix(training_size);
  SampleMatrix test_set=c10.getTestMatrix(test_size);

  normalizeSet(training_set);
  normalizeSet(test_set);

  MBMLP mbmlp(training_set,test_set,
              learning_rate,training_set.vectors.rows(),
              batch_size);
  std::cout<<"HI"<<std::endl;
  mbmlp.randomInit();
  std::cout<<"HI"<<std::endl;


  EventTimer et;
  // Create log file
  ensure_a_path_exists(store_path);
  std::ofstream log(store_path+"/log.csv");
  if(!log.is_open()){
    std::cerr<<"Error in opening: "<<store_path+"/log.csv"<<std::endl;
    exit(1);
  }
  log<<"epoch,train_accuracy,train_hinge_loss,test_accuracy,test_hinge_loss"<<"\n";

  for(int starting_epoch=0;starting_epoch<epoch_number;starting_epoch+=epoch_step){
    // Train for a bunch of epochs
    et.start("Batch train");
    mbmlp.runEpochBatch(epoch_step);
    et.stop();
    // Test on training  
    float train_accuracy,train_hinge_loss;
    mbmlp.testOnSet(training_set, train_accuracy, train_hinge_loss);
    std::cout<<"Train: "<<train_accuracy<<" "<<train_hinge_loss<<std::endl;
    // Test on set 
    float test_accuracy,test_hinge_loss;
    mbmlp.testOnSet(test_set, test_accuracy, test_hinge_loss);
    std::cout<<"Test: "<<test_accuracy<<" "<<test_hinge_loss<<std::endl;
    log<<starting_epoch+epoch_step-1<<","<<train_accuracy<<","<<train_hinge_loss<<","
      <<test_accuracy<<","<<test_hinge_loss<<"\n";
  }



  log.close();
  et.displayIntervals();

  
  script_et.stop();
  script_et.displayIntervals();


  
  return 0;

}
