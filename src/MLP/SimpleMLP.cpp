#include "MLP/SimpleMLP.hpp"

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
  bias=VectorXf(input_size);
  output=MatrixXf(1,batch_size);
}




void randomInit();

// Classic methods
void forwardPass(const MatrixXf& input);
void getHingeLoss(const VectorXi& labels);
void backwardPass(const MatrixXf& input,
                  const VectorXi& labels);

void runEpoch();
void testOnSet(const SampleMatrix& set,
               float& accuracy,
               float& hinge_loss);

void store();
void load();
