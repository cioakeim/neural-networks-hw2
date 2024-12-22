#include "MLP/SimpleMLP.hpp"
#include <random>

SimpleMLP::SimpleMLP(const float learning_rate,
                     int input_size,
                     int batch_size,
                     const SampleMatrix& training_set,
                     const SampleMatrix& test_set):
  learning_rate(learning_rate),
  input_size(input_size),
  batch_size(batch_size),
  training_set(training_set),
  test_set(test_set){

  weights=MatrixXf(1,input_size);
  bias=VectorXf(1);
  output=MatrixXf(1,batch_size);
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


void getHingeLoss(const VectorXi& labels);
void backwardPass(const MatrixXf& input,
                  const VectorXi& labels);

void runEpoch();
void testOnSet(const SampleMatrix& set,
               float& accuracy,
               float& hinge_loss);

void store();
void load();
