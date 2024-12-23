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
  std::string store_path="/data/SimpleMLP/1v1/test";

  int training_size=50e3;
  int test_size=10e3;
  int class1_id=0;
  int class2_id=1;


  // Loading dataset...
  Cifar10Handler c10(dataset_path); 

  std::cout<<"Loading dataset..."<<std::endl;
  std::vector<SamplePoint> training_set=c10.getTrainingList(training_size);
  std::vector<SamplePoint> test_set=c10.getTestList(test_size);

  

}
