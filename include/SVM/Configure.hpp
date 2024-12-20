#ifndef CONFIGURE_HPP 
#define CONFIGURE_HPP

#include <string>
#include <vector>
#include "SVM/Kernels.hpp"

enum KernelType{LINEAR,POLY,RBF};

/**
 * @brief Full config for experiment run.
 *
 * Contains dataset info as well as SVM config info.
 * Used only for 2-class SVM (not multiclass)
*/
struct SVM2ClassConfig{
  // Where to store the results
  std::string store_path;
  // For dataset
  std::string dataset_path;
  int training_size;
  int test_size;

  // Classes chosen
  int class1_id;
  int class2_id;

  // For kernel
  KernelType kernel_type;
  KernelParameters kernel_parameters; 

  // For C parameter there is a vector
  std::vector<float> C_list;
};

void configureFromArguments(int argc,char* argv[],
                            SVM2ClassConfig& config);

void storeConfigInfo(const SVM2ClassConfig& config,
                     std::string file_path);


#endif
