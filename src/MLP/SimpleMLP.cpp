#include "MLP/SimpleMLP.hpp"
#include "CommonLib/basicFuncs.hpp"
#include <random>
#include <iostream>

SimpleMLP::SimpleMLP(const float learning_rate,
                     const int input_size,
                     const int batch_size,
                     const int class1,const int class2,
                     const SampleMatrix& training_set,
                     const SampleMatrix& test_set):
  learning_rate(learning_rate),
  input_size(input_size),
  batch_size(batch_size),
  class1(class1),class2(class2),
  full_training_set(training_set),
  full_test_set(test_set){

  weights=VectorXf(input_size);
  output=VectorXf(batch_size);
  delta=VectorXf(batch_size);
}


void SimpleMLP::loadDataset(){
  training_set=extract1v1Dataset(full_training_set, class1, class2);
  test_set=extract1v1Dataset(full_test_set, class1, class2);
}


void SimpleMLP::randomInit(){
  const int size=weights.size();
  const float stddev= std::sqrt(2.0f/size);
  // Init rng 
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0,stddev);

  for(int i=0;i<size;i++){
    weights(i)=(dist(gen));
  }
  bias=(dist(gen));
}


// Classic methods
void SimpleMLP::forwardPass(const MatrixXf& input){
  const VectorXf u=(weights.transpose()*input).array()+bias;
  output=u;  
}

float SimpleMLP::getHingeLoss(const VectorXi& labels){
  const E::VectorXf margin=1-labels.cast<float>().array()*output.array();
  return margin.cwiseMax(0.0f).array().mean();
}


void SimpleMLP::backwardPass(const MatrixXf& input,
                             const VectorXi& labels){
  const float lambda=1e-7;
  // Get local error 
  const E::VectorXf margin=1-(output.cwiseProduct(labels.cast<float>())).array();
  const E::VectorXf indicator=(margin.array()>0).cast<float>();


  delta=indicator.cwiseProduct(-labels.cast<float>());

  // Since no other layers update on the spot 
  weights-=input*delta*(learning_rate/batch_size)+lambda*weights;
  bias-=delta.sum()*(learning_rate/batch_size)+lambda*bias;
  //std::cout<<"Weight range: "<<weights.minCoeff()<<","<<weights.maxCoeff()<<std::endl;
}


void SimpleMLP::runEpoch(){
  std::cout<<"Shufflin.."<<std::endl;
  shuffleDatasetInPlace(training_set);
  std::cout<<"Done."<<std::endl;
  const int training_size=training_set.vectors.cols();
  for(int idx=0;idx<training_size;idx+=batch_size){
    const int final=(idx+batch_size)<=training_size ? batch_size : training_size-idx;
    const MatrixXf& input=training_set.vectors.middleCols(idx,final);
    const VectorXi& labels=training_set.labels.segment(idx,final);
    forwardPass(input);
    backwardPass(input, labels);
  }
  std::cout<<"Epoch done."<<std::endl;
}




void SimpleMLP::testOnSet(const SampleMatrix& set,
                          float& accuracy,
                          float& hinge_loss){
  forwardPass(set.vectors);
  E::VectorXi pred=2*((output.array()>0).cast<int>()).array()-1;
  accuracy=(pred.array()==set.labels.array()).cast<float>().mean();
  hinge_loss=getHingeLoss(set.labels);
}

void store();
void load();
