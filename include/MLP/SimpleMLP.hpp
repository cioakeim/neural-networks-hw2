#ifndef SIMPLE_MLP_HPP
#define SIMPLE_MLP_HPP

#include <string>
#include <Eigen/Dense>
#include "CommonLib/basicStructs.hpp"

namespace E=Eigen;
using E::MatrixXf;
using E::VectorXf;
using E::VectorXi;
using VectorFunction = std::function<MatrixXf(const MatrixXf)>;

/**
 * @brief An implementation of 1v1 1 Layer MLP
 *
 * Uses hinge loss.
 */
class SimpleMLP{
private:
  // For I/O
  std::string store_path; 
  // Weights and bias
  MatrixXf weights;
  VectorXf bias;
  MatrixXf output;
  
  // Parameters
  VectorFunction activation_function;
  VectorFunction activation_derivative;
  const float learning_rate;
  const int input_size;
  const int batch_size;

  // Dataset
  const SampleMatrix& training_set;
  const SampleMatrix& test_set;

  
public:
  SimpleMLP(const float learning_rate,
            int input_size,
            int batch_size,
            const SampleMatrix& training_set,
            const SampleMatrix& test_set);


  void setStorePath(std::string path){this->store_path=path;}
  void setFunction(VectorFunction f,VectorFunction f_dot){
    this->activation_function=f;
    this->activation_derivative=f_dot;
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
};


#endif
