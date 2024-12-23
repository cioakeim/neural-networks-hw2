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
  //svm->clearSolution();
}




void MultiClassSVM::trainAllSVMs(float& train_hinge_loss,
                                 float& test_hinge_loss){
  train_hinge_loss=test_hinge_loss=0;
  for(int class_1=0;class_1<CLASS_NUMBER-1;class_1++){
    for(int class_2=class_1+1;class_2<CLASS_NUMBER;class_2++){
      std::cout<<"Training pair: ("<<class_1<<","<<class_2<<")"<<std::endl;
      // Train the actual svm
      trainTwoClassSVM(class_1,class_2);
      // Get hinge loss on training and on set
      SVM* svm=two_class_svms[pair_to_svm_lut[class_1][class_2]];
      // For training 
      std::cout<<"Testing"<<std::endl;
      const E::VectorXf tr_out=svm->output(svm->getTrainingSetRef().vectors);
      std::cout<<"Train"<<std::endl;
      float hinge_loss=svm->getHingeLoss(tr_out,
                                         svm->getTrainingSetRef().labels);
      std::cout<<"Hinge loss: "<<hinge_loss<<std::endl;
      train_hinge_loss+=hinge_loss;
      // For testing
      const E::VectorXf te_out=svm->output(svm->getTestSetRef().vectors);
      std::cout<<"Test"<<std::endl;
      hinge_loss=svm->getHingeLoss(te_out,
                                   svm->getTestSetRef().labels);
      std::cout<<"Test Hinge loss: "<<hinge_loss<<std::endl;
      test_hinge_loss+=hinge_loss;

      float accuracy;
      svm->testOnSet(svm->getTrainingSetRef(), accuracy, hinge_loss);
      std::cout<<"TRAIN ACCURACY: "<<accuracy<<std::endl;

      std::cout<<"Done"<<std::endl;
      svm->clearDataset();
      svm->clearSupportVectors();
    }
  }
  int svm_num=SVM_NUMBER;
  train_hinge_loss/=static_cast<float>(svm_num);
  test_hinge_loss/=static_cast<float>(svm_num);
}


void MultiClassSVM::testOnSet(const SampleMatrix& set,
                              float& accuracy){
  const int sample_size=set.labels.size();
  // For final mean hinge loss
  // In the below matrices, each column is for the prediction of 1 sample
  // Each SVM will place its vote here.
  E::MatrixXf votes=E::MatrixXf::Zero(CLASS_NUMBER,sample_size);
  
  // In the case of a tie, the mean hinge losses are kept here
  E::MatrixXf total_confidence=E::MatrixXf::Zero(CLASS_NUMBER,sample_size);

  // Each binary SVM votes and places its confidence value in the corresponding bin
  for(int class_1=0;class_1<CLASS_NUMBER-1;class_1++){
    for(int class_2=class_1+1;class_2<CLASS_NUMBER;class_2++){
      SVM* svm=two_class_svms[pair_to_svm_lut[class_1][class_2]];
      // If support vectors were cleared bring them back
      if(svm->areSupportVectorsEmpty()){
        svm->constructDatasetFromClassSets();
        svm->loadSupportVectors();
      }

      // Get output of svm
      E::VectorXf output=svm->output(set.vectors);
      // Get prediction
      E::VectorXi prediction=svm->predictSet(output);
      if(((prediction.array()!=-1)&&(prediction.array()!=1)).sum()>0){
        std::cout<<"Error to pred"<<std::endl;
      }
      // Convert output to confidence
      // Convert to binary representation
      prediction=(prediction.array()+1)/2;
      if(((prediction.array()!=0)&&(prediction.array()!=1)).sum()>0){
        std::cout<<"Error to binary"<<std::endl;
      }
      // Convert to class ids
      prediction=(prediction.array()*class_1+(1-prediction.array())*class_2);
      if(((prediction.array()<0)||(prediction.array()>9)).sum()>0){
        std::cout<<"Error to class_id"<<std::endl;
      }
      // For each prediction place in correct bin
      //#pragma omp parallel for
      output=output.array().abs();
      for(int i=0;i<sample_size;i++){
        votes(prediction(i),i)+=output(i);
        total_confidence(prediction(i),i)+=output(i);
      }
    }
  }
  // Time for voting results
  E::VectorXi final_prediction(sample_size);

  // Reduce column-wise
  for(int col=0;col<sample_size;col++){
    E::VectorXf::Index idx;
    votes.col(col).maxCoeff(&idx);
    final_prediction(col)=idx;
  }
  // Get final prediction
  accuracy=(final_prediction.array()==set.labels.array()).cast<float>().mean();
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
