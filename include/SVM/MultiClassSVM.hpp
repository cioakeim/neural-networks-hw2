#ifndef MULTI_CLASS_SVM_HPP
#define MULTI_CLASS_SVM_HPP

#include "SVM/SVM.hpp"
#include "CommonLib/basicStructs.hpp"
#include <string>

#define CLASS_NUMBER 10
#define SVM_NUMBER ((CLASS_NUMBER)*(CLASS_NUMBER-1)/2)


/**
 * @brief Multiple 1v1 SVMs used for 10 class classification
 */
class MultiClassSVM{
private:
  // Name and filepath of SVM
  std::string name;
  std::string root_filepath;

  // The dataset
  const SampleMatrix& training_set;
  const SampleMatrix& test_set;
  // Training set split to separate samples
  std::vector<E::MatrixXf> train_class_samples;
  std::vector<E::MatrixXf> test_class_samples;

  // The actual 2-class SVMs 
  int pair_to_svm_lut[CLASS_NUMBER][CLASS_NUMBER];
  SVM* two_class_svms[SVM_NUMBER];

public:
  MultiClassSVM(const SampleMatrix& training_set,
                const SampleMatrix& test_set);
  ~MultiClassSVM();

  // Setters / Getters (only ones needed are implemented)
  void setNameAndPath(std::string name,
                      std::string root_filepath){
    this->name=name;
    this->root_filepath=root_filepath;
  };

  SVM* getSVMPointer(int class_1_idx,int class_2_idx){
    return two_class_svms[pair_to_svm_lut[class_1_idx][class_2_idx]];
  }

  void storeSVM();
  void loadSVM();

  void setKernelToAll(std::function<E::MatrixXf(const E::MatrixXf,
                                                const E::MatrixXf,
                                                KernelParameters)>func,
                      KernelParameters Kernel_parameters);

  void setCToAll(float C);

  void trainTwoClassSVM(int class_1_idx,int class_2_idx);

  void testTwoClassSVM(int class_1_idx,int class_2_idx,
                       const SampleMatrix& test_set,
                       float& accuracy,
                       float& mean_hinge_loss);



  void trainAllSVMs();
};




#endif
