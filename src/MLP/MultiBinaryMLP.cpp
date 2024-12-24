#include "MLP/MultiBinaryMLP.hpp"
#include "CommonLib/basicFuncs.hpp"

MBMLP::MBMLP(const SampleMatrix& training_set,
             const SampleMatrix& test_set,
             const float learning_rate,
             int input_size,
             int batch_size):
  training_set(training_set),
  test_set(test_set){
  train_class_sets=splitDataset(training_set,CLASS_NUMBER);
  test_class_sets=splitDataset(test_set,CLASS_NUMBER);
  for(int class1=0;class1<CLASS_NUMBER;class1++){
    for(int class2=class1+1;class2<CLASS_NUMBER;class2++){
      mlps[pair_to_mlp[class1][class2]]=new SimpleMLP(learning_rate,
                                                      input_size,
                                                      batch_size,
                                                      class1,class2,
                                                      training_set,
                                                      test_set);
    }
  }
}

MBMLP::~MBMLP(){
  for(int i=0;i<MLP_NUMBER;i++){
    delete mlps[i];
  }
}

void MBMLP::setPath(std::string filepath){
  this->filepath=filepath;
  for(int i=0;i<MLP_NUMBER;i++){
    
  }
}


void storeMLP();
void loadMLP();

void randomInit();
void loadDatasets();
void runEpoch();

float trainHingeLoss();
float testHingeLoss();

void testOnSet(const SampleMatrix& set,
               float& accuracy);
