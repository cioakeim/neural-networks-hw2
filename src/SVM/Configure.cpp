#include "SVM/Configure.hpp"
#include <iostream>
#include <fstream>
#include "CommonLib/basicFuncs.hpp"

static float stringToFloat(const std::string& str) {
  try {
    return std::stof(str);
  } catch (const std::invalid_argument& e) {
    throw std::runtime_error("Invalid argument: cannot convert string to float: " + str);
  }
}


static std::string kernel_to_string(KernelType type){
  switch(type){
    case LINEAR:
      return "Linear";
    break;
    case POLY:
      return "Poly";
    break;
    case RBF:
      return "RBF";
    break;
  }
  return "";
}


static int stringToInt(const std::string& str) {
  try {
    return std::stoi(str);
  } catch (const std::invalid_argument& e) {
    throw std::runtime_error("Invalid argument: cannot convert string to int: " + str);
  } catch (const std::out_of_range& e) {
    throw std::runtime_error("Out of range: cannot convert string to int: " + str);
  }
}


static std::vector<float> commaSeparatedToVector(const std::string& str) {
  std::vector<float> result;
  std::stringstream ss(str);
  std::string item;

  while (std::getline(ss, item, ',')) {
    try {
      result.push_back(std::stof(item));
    } catch (const std::invalid_argument& e) {
      throw std::runtime_error("Invalid argument in list: cannot convert to float: " + item);
    }
  }
  return result;
}


void configureFromArguments(int argc,char* argv[],
                            SVM2ClassConfig& config){
  // Convert arguments to strings
  std::string exec_name = argv[0]; // Name of the current exec program
  std::vector<std::string> args;
  if (argc > 1) {
    args.assign(argv + 1, argv + argc);
  }

  if(argc==1){
    std::cout<<"Usage: "+exec_name<<" [store_path] [dataset_path] [training_size] "
    "[test_size] [class1_id] [class2_id] [C_list] [kernel_type] [kernel_parameters]"
      <<std::endl;

    std::cout<<"Arguments:\n\tstore_path,dataset_path: string (path) \n"
      "\ttraining_size,test_size: positive integers (Max values:50000,10000) \n" 
      "\tclass2_id,class2_id: integers in {0,..,9}\n "
      "\tC_list: comma separated floats for parameter sweep (e.g: 0.1,1,10)"
      "\tkernel_type: enum{LINEAR,POLY,RBF} \n"
      "\tkernel_parameters:\n \t\t POLY: [d] [c] (degree and constant) \n "
      "\t\t RBF: [sigma] (sigma constant AFTER NORMALIZATION WITH FEATURE SIZE) \n"
      "\n\nTo run with default arguments: "+exec_name+" -d"
      <<std::endl;
    exit(1);
  }
  if(argc==2 && (args[0]=="-d")){
    std::cout<<"Running with default arguments"<<std::endl;
    return;
  }
  int z=0;
  config.store_path=args[z++];
  config.dataset_path=args[z++];
  config.training_size=stringToInt(args[z++]);
  config.test_size=stringToInt(args[z++]);
  config.class1_id=stringToInt(args[z++]);
  config.class2_id=stringToInt(args[z++]);
  config.C_list=commaSeparatedToVector(args[z++]);
  // For kernel type:
  if(args[z]=="LINEAR"){
    config.kernel_type=LINEAR; 
  }
  else if(args[z]=="POLY"){
    z++;
    config.kernel_type=POLY;
    config.kernel_parameters.poly_d=stringToInt(args[z++]);
    config.kernel_parameters.poly_c=stringToFloat(args[z++]);
  }
  else if(args[z]=="RBF"){
    z++;
    config.kernel_type=RBF;
    config.kernel_parameters.rbf_sigma=stringToFloat(args[z++]);
  }
  return;
}


void storeConfigInfo(const SVM2ClassConfig& config,
                     std::string file_path){
  ensure_a_path_exists(file_path);
  std::ofstream file(file_path+"/config_info.csv");
  if(!file.is_open()){
    std::cerr<<"Error in opening: "<<file_path+"/config_info.csv"<<std::endl;
    exit(1);
  }
  file<<"Training size: "<<config.training_size<<
    "\nTest size: "<<config.test_size<<
    "\nClass pair: ("<<config.class1_id<<","<<config.class2_id<<")"<<
    "\nKernel type: "<< kernel_to_string(config.kernel_type)<<
    "\nD: "<<config.kernel_parameters.poly_d<<
    "\nC: "<<config.kernel_parameters.poly_c<<
    "\nSigma: "<<config.kernel_parameters.rbf_sigma<<"\n";
  file.close(); 
}
