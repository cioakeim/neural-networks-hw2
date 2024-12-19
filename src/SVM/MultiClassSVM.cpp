#include "SVM/MultiClassSVM.hpp"
#include "CommonLib/basicFuncs.hpp"
#include "SVM/SVM.hpp"
#include <cinttypes>
#include <string>
#include <iostream>

// Split the main training set to separate class sets
std::vector<E::MatrixXf> splitDataset(const SampleMatrix& training_set){
  std::vector<std::vector<int>> class_idx(CLASS_NUMBER);
  // Map indices to index vectors
  for(int i=0;i<training_set.labels.size();i++){
    class_idx[training_set.labels(i)].push_back(i);
  }
  // Create actual matrices
  std::vector<E::MatrixXf> result(CLASS_NUMBER);
  #pragma omp parallel for
  for(int i=0;i<CLASS_NUMBER;i++){
    const int samples_num=class_idx[i].size();
    result[i]=E::MatrixXf(training_set.vectors.rows(),samples_num);
    for(int j=0;j<samples_num;j++){
      result[i].col(i)=training_set.vectors.col(class_idx[i][j]);
    }
  }
  return result;
}




MultiClassSVM::MultiClassSVM(const SampleMatrix& training_set,
                             const SampleMatrix& test_set):
  training_set(training_set),
  test_set(test_set){
  this->train_class_samples=splitDataset(training_set);
  std::cout<<"Train class size: "<<this->train_class_samples.size()<<std::endl;
  this->test_class_samples=splitDataset(test_set);
  std::cout<<"Test class size: "<<this->test_class_samples.size()<<std::endl;
  // Init LUT and SVMs
  int svm_idx=0;
  for(int i=0;i<CLASS_NUMBER;i++){
    for(int j=0;j<CLASS_NUMBER;j++){
      this->pair_to_svm_lut[i][j]=-1;
    }
  }
  for(int class_1=0;class_1<CLASS_NUMBER-1;class_1++){
    for(int class_2=class_1+1;class_2<CLASS_NUMBER;class_2++){
      //std::cout<<"Config: "<<class_1<<","<<class_2<<" with svm_idx: "<<svm_idx<<std::endl;
      this->pair_to_svm_lut[class_1][class_2]=svm_idx;
      //std::cout<<"Assigned: "<<this->pair_to_svm_lut[class_1][class_2]<<std::endl;
      this->two_class_svms[svm_idx]=new SVM(train_class_samples[class_1],
                                            test_class_samples[class_1],
                                            train_class_samples[class_2],
                                            test_class_samples[class_2]);
      svm_idx++;
    }
  }
}


MultiClassSVM::~MultiClassSVM(){
  for(int i=0;i<SVM_NUMBER;i++){
    delete two_class_svms[i];
  }
}


void MultiClassSVM::setKernelToAll(std::function<E::MatrixXf(const E::MatrixXf,
                                                             const E::MatrixXf,
                                                             KernelParameters)>func,
                                   KernelParameters kernel_parameters){
  //std::cout<<"SVM_NUMBER: "<<SVM_NUMBER<<std::endl;
  for(int i=0;i<SVM_NUMBER;i++){
    two_class_svms[i]->setKernelFunction(func,kernel_parameters);
  }
}


void MultiClassSVM::setCToAll(float C){
  for(int i=0;i<SVM_NUMBER;i++){
    two_class_svms[i]->setC(C);
  }
}


void MultiClassSVM::trainTwoClassSVM(int class_1_idx,int class_2_idx){
  /**
  for(int i=0;i<CLASS_NUMBER;i++){
    for(int j=0;j<CLASS_NUMBER;j++){
      std::cout<<this->pair_to_svm_lut[i][j]<<" ";
    }
    std::cout<<"\n"<<std::endl;
  }
  */
  // Select svm using LUT
  //std::cout<<"Here"<<std::endl;
  //std::cout<<"Indices: "<<class_1_idx<<","<<class_2_idx<<std::endl;
  //std::cout<<"Svm idx: "<<pair_to_svm_lut[class_1_idx][class_2_idx]<<std::endl;
  SVM* svm=two_class_svms[pair_to_svm_lut[class_1_idx][class_2_idx]];

  //std::cout<<"Construct"<<std::endl;
  svm->constructDatasetFromClassSets();
  //std::cout<<"Solve"<<std::endl;
  svm->solveAndStore();
  //std::cout<<"Clear"<<std::endl;
  svm->clearDataset();
  svm->clearSolution();
}

void MultiClassSVM::testTwoClassSVM(int class_1_idx,int class_2_idx,
                                    const SampleMatrix& test_set,
                                    float& accuracy,
                                    float& mean_hinge_loss){
  SVM* svm=two_class_svms[pair_to_svm_lut[class_1_idx][class_2_idx]];
  svm->testOnSet(test_set,accuracy,mean_hinge_loss);
}


void MultiClassSVM::trainAllSVMs(){
  #pragma omp parallel for
  for(int class_1=0;class_1<CLASS_NUMBER-1;class_1++){
    #pragma omp parallel for
    for(int class_2=class_1+1;class_2<CLASS_NUMBER;class_2++){
      std::cout<<"Training pair: ("<<class_1<<","<<class_2<<")"<<std::endl;
      trainTwoClassSVM(class_1,class_2);
      std::cout<<"End of training: "<<pair_to_svm_lut[class_1][class_2]<<std::endl;
    }
  }
}


void MultiClassSVM::storeSVM(){
  std::string svm_path=root_filepath+"/"+name;
  ensure_a_path_exists(svm_path);
  for(int i=0;i<SVM_NUMBER;i++){
    std::string two_class_path=svm_path+"/SVM_"+std::to_string(i);
    ensure_a_path_exists(two_class_path);
    two_class_svms[i]->storeToFile(two_class_path);
  }
}


void MultiClassSVM::loadSVM(){
  std::string svm_path=root_filepath+"/"+name;
  for(int i=0;i<SVM_NUMBER;i++){
    two_class_svms[i]->loadFromFile(svm_path);
  }
  
}
