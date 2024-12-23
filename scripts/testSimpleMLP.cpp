#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

#include "CommonLib/cifarHandlers.hpp"
#include "CommonLib/basicFuncs.hpp"
#include "CommonLib/LogHandler.hpp"
#include "MLP/ActivationFunctions.hpp"
#include "MLP/SimpleMLP.hpp"
#include "CommonLib/eventTimer.hpp"
#include <time.h>
#include <csignal>


int main(int argc,char* argv[]){
  std::string dataset_path="../data/cifar-10-batches-bin";
  std::string store_path="../data/SimpleMLP/1v1/test";

  int training_size=50e3;
  int test_size=10e3;
  int class1_id=0;
  int class2_id=1;

  float learning_rate=1e-2;
  int input_size=3072;
  int batch_size=1000;
  int epoch_number=200;


  EventTimer total_et;
  total_et.start("Script execution time");
  // Loading dataset...
  Cifar10Handler c10(dataset_path); 

  std::cout<<"Loading dataset..."<<std::endl;
  SampleMatrix full_training_set=c10.getTrainingMatrix(training_size);
  SampleMatrix full_test_set=c10.getTestMatrix(test_size);

  SampleMatrix training_set=extract1v1Dataset(full_training_set, class1_id, class2_id);
  SampleMatrix test_set=extract1v1Dataset(full_test_set, class1_id, class2_id);

  normalizeSet(training_set);
  normalizeSet(test_set);

  std::cout<<"Dataset extraction successful.."<<std::endl;


  std::cout<<"Constructing mlp"<<std::endl;
  SimpleMLP mlp=SimpleMLP(learning_rate,input_size,batch_size,
                          training_set,test_set);
  std::cout<<"Success"<<std::endl;




  mlp.randomInit();
  mlp.setFunction(linear, linearder);
  mlp.setStorePath(store_path);

  // Create log file
  ensure_a_path_exists(store_path);
  std::ofstream log(store_path+"/log.csv");
  if(!log.is_open()){
    std::cerr<<"Error in opening: "<<store_path+"/log.csv"<<std::endl;
    exit(1);
  }
  log<<"epoch,train_accuracy,train_hinge_loss,test_accuracy,test_hinge_loss"<<"\n";

  for(int epoch=0;epoch<epoch_number;epoch++){
    EventTimer et;
    std::cout<<"Epoch number: "<<epoch<<std::endl;
    std::string epoch_str=std::to_string(epoch);
    
    et.start("Train epoch "+epoch_str);
    mlp.runEpoch();
    et.stop();

    float train_accuracy,train_hinge_loss;
    et.start("Train test epoch "+epoch_str);
    mlp.testOnSet(training_set, train_accuracy, train_hinge_loss);
    et.stop();

    float test_accuracy,test_hinge_loss;
    et.start("Test test epoch "+epoch_str);
    mlp.testOnSet(test_set, test_accuracy, test_hinge_loss);
    et.stop();

    std::cout<<"Results of epoch:\nTrain: Hinge loss: "<<train_hinge_loss
      <<" Accuracy: "<<train_accuracy
      <<"\nTest: Hinge loss: "<<test_hinge_loss<<" Accuracy: "<<test_accuracy
      <<std::endl;

    et.displayIntervals();
    et.writeToFile(store_path+"/epoch_"+epoch_str+"_time.txt");
    et.clearEvents();
  }
  log.close();

  
  total_et.stop();
  total_et.displayIntervals();
}
