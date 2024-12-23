#include "MLP/SimpleMLP.hpp"
#include "CommonLib/basicFuncs.hpp"
#include <random>

SimpleMLP::SimpleMLP(const float learning_rate,
                     int input_size,
                     int batch_size,
                     SampleMatrix& training_set,
                     SampleMatrix& test_set):
  learning_rate(learning_rate),
  input_size(input_size),
  batch_size(batch_size),
  training_set(training_set),
  test_set(test_set){

  weights=MatrixXf(1,input_size);
  bias=VectorXf(1);
  output=MatrixXf(1,batch_size);
  delta=MatrixXf(1,batch_size);
}


void SimpleMLP::randomInit(){
  const int rows=weights.rows();
  const int cols=weights.cols();
  const float stddev= std::sqrt(2.0f/rows);
  // Init rng 
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0,stddev);

  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++){
      weights(i,j)=(dist(gen));
    }
    bias(i)=(dist(gen));
  }
}


// Classic methods
void SimpleMLP::forwardPass(const MatrixXf& input){
  output=activation_function((weights*input).colwise()+bias);  
}

float SimpleMLP::getHingeLoss(const VectorXi& labels){
  return (1-(labels.array()).cast<float>()*output.array()).cwiseMax(0).mean();
}


void SimpleMLP::backwardPass(const MatrixXf& input,
                             const VectorXi& labels){
  // Get local error 
  delta=((1-(output.cwiseProduct(labels.cast<float>())).array()).array()>0).cast<float>();
  const E::VectorXf temp=-labels.cast<float>().array()*activation_derivative(output).array();
  delta.cwiseProduct(temp);

  // Since no other layers update on the spot 
  weights-=input*delta.transpose()*(learning_rate/batch_size);
  bias(0,0)-=delta.mean()*learning_rate;
}


void SimpleMLP::runEpoch(){
  shuffleDatasetInPlace(training_set);
  const int training_size=training_set.vectors.cols();
  for(int idx=0;idx<training_size;idx+=batch_size){
    const MatrixXf& input=training_set.vectors.middleCols(idx,batch_size);
    const VectorXi& labels=training_set.labels.segment(idx,batch_size);
    forwardPass(input);
    backwardPass(input, labels);
  }
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
