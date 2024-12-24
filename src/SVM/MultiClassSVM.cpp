#include "SVM/MultiClassSVM.hpp"
#include "CommonLib/basicFuncs.hpp"
#include "SVM/SVM.hpp"
#include <cmath>
#include <string>
#include <iostream>

static void NaNcheck(const E::MatrixXf& mat,std::string label){
  if(mat.array().isNaN().cast<int>().sum()>0){
    std::cerr<<label<<std::endl;
    exit(1);
  }
}





MultiClassSVM::MultiClassSVM(const SampleMatrix& training_set,
                             const SampleMatrix& test_set):
  training_set(training_set),
  test_set(test_set){

  this->train_class_samples=splitDataset(training_set,CLASS_NUMBER);
  this->test_class_samples=splitDataset(test_set,CLASS_NUMBER);
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

void MultiClassSVM::setPath(std::string filepath){
  this->filepath=filepath;
  ensure_a_path_exists(filepath);
  for(int i=0;i<SVM_NUMBER;i++){
    std::string two_class_path=filepath+"/SVM_"+std::to_string(i);
    ensure_a_path_exists(two_class_path);
    two_class_svms[i]->setFolderPath(two_class_path);
  }
};


void  MultiClassSVM::getTotalSVStats(int& sum,
                                     float& mean,
                                     float& sigma){
  E::VectorXf sv_cnts(SVM_NUMBER);
  for(int i=0;i<SVM_NUMBER;i++)
    sv_cnts(i)=static_cast<float>(two_class_svms[i]->getSVCount());
  sum=sv_cnts.cast<int>().sum();
  mean=sv_cnts.mean();
  sigma=std::sqrt(sv_cnts.array().pow(2).mean());
}


void MultiClassSVM::setKernelToAll(std::function<E::MatrixXf(const E::MatrixXf,
                                                             const E::MatrixXf,
                                                             KernelParameters)>func,
                                   KernelParameters kernel_parameters){
  for(int i=0;i<SVM_NUMBER;i++){
    two_class_svms[i]->setKernelFunction(func,kernel_parameters);
  }
}


void MultiClassSVM::setCToAll(float C){
  for(int i=0;i<SVM_NUMBER;i++){
    two_class_svms[i]->setC(C);
  }
}



void MultiClassSVM::trainAllSVMs(std::vector<float> C_list){
  for(int class_1=0;class_1<CLASS_NUMBER-1;class_1++){
    for(int class_2=class_1+1;class_2<CLASS_NUMBER;class_2++){
      SVM* svm=two_class_svms[pair_to_svm_lut[class_1][class_2]];
      svm->constructDatasetFromClassSets();
      svm->computeKernelMatrix();
      svm->configLinearCostTerm();
      for(auto c: C_list){
        svm->setC(c);
        svm->configConstraints();
        svm->solveQuadraticProblem();
        svm->storeSVIndicesAndAVector();
        svm->storeToFile();
        svm->clearWholeSolution();
      }
      svm->clearDataset();
      svm->deconstructQPProblem();
    }
  }
}


void MultiClassSVM::loadSVMs(const float c){
  for(int i=0;i<SVM_NUMBER;i++){
    SVM* svm=two_class_svms[i];
    svm->constructDatasetFromClassSets();
    svm->clearWholeSolution();
    svm->setC(c);
    svm->loadFromFile();
    svm->loadSupportVectors();
    svm->clearDataset();
  }
}


void MultiClassSVM::testOnSet(const SampleMatrix& set,
                              float& accuracy,
                              float& hinge_loss){
  EventTimer my_et;
  int hinge_denominator=0;
  const int sample_size=set.labels.size();
  // For final mean hinge loss
  // In the below matrices, each column is for the prediction of 1 sample
  // Each SVM will place its vote here.
  E::MatrixXf votes=E::MatrixXf::Zero(CLASS_NUMBER,sample_size);
  
  // In the case of a tie, the mean hinge losses are kept here

  my_et.start("Phase1");
  // Each binary SVM votes and places its confidence value in the corresponding bin
  for(int class_1=0;class_1<CLASS_NUMBER-1;class_1++){
    for(int class_2=class_1+1;class_2<CLASS_NUMBER;class_2++){
      SVM* svm=two_class_svms[pair_to_svm_lut[class_1][class_2]];
      // Get output of svm
      const E::VectorXf output=svm->output(set.vectors);
      // Get prediction
      const E::VectorXi prediction=svm->predictSet(output);
      // Convert output to confidence
      // Convert to binary representation
      const E::VectorXi pred_class_bin=(prediction.array()+1)/2;
      // Convert to class ids
      const E::VectorXi pred_class_id=(pred_class_bin.array()*class_1+(1-pred_class_bin.array())*class_2);
      // For each prediction place in correct bin
      #pragma omp parallel for
      for(int i=0;i<sample_size;i++){
        votes(pred_class_id(i),i)+=std::abs(output(i));
      }
      // For hinge loss
      const E::VectorXi bin_labels=(set.labels.array()==class_1).array().cast<int>()-
                                   (set.labels.array()==class_2).array().cast<int>();
      hinge_loss+=(1-output.array()*bin_labels.cast<float>().array()).cwiseMax(0).sum();
      hinge_denominator+=(bin_labels.array()!=0).cast<int>().sum();
    }
  }
  my_et.stop();

  my_et.start("Phase2");
  // Time for voting results
  E::VectorXi final_prediction(sample_size);

  // Reduce column-wise
  #pragma omp parallel for
  for(int col=0;col<sample_size;col++){
    E::VectorXf::Index idx;
    votes.col(col).maxCoeff(&idx);
    final_prediction(col)=idx;
  }
  // Get final prediction
  accuracy=(final_prediction.array()==set.labels.array()).cast<float>().mean();
  hinge_loss=hinge_loss/hinge_denominator;
  my_et.stop();
  my_et.displayIntervals();
}



void MultiClassSVM::storeSVM(){
  for(int i=0;i<SVM_NUMBER;i++){
    two_class_svms[i]->storeToFile();
  }
}


void MultiClassSVM::loadSVM(){
  for(int i=0;i<SVM_NUMBER;i++){
    two_class_svms[i]->loadFromFile();
  }
  
}
