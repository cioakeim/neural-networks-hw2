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
  VectorXf weights;
  float bias;
  VectorXf output;
  VectorXf delta;
  
  // Parameters
  VectorFunction activation_function;
  VectorFunction activation_derivative;
  const float learning_rate;
  const int input_size;
  const int batch_size;


  // Ref to full stuff
  const SampleMatrix& full_training_set;
  const SampleMatrix& full_test_set;
  const int class1,class2;

  // Dataset
  SampleMatrix training_set;
  SampleMatrix test_set;


  
public:
  SimpleMLP(const float learning_rate,
            const int input_size,
            const int batch_size,
            const int class1,const int class2,
            const SampleMatrix& training_set,
            const SampleMatrix& test_set);


  void setStorePath(std::string path){this->store_path=path;}
  void setFunction(VectorFunction f,VectorFunction f_dot){
    this->activation_function=f;
    this->activation_derivative=f_dot;
  }
  void loadDataset();
  void clearDataset(){
    training_set.vectors.conservativeResize(0,0);
    training_set.labels.conservativeResize(0);
    test_set.vectors.conservativeResize(0,0);
    test_set.labels.conservativeResize(0);
  }

  void randomInit();

  // Classic methods
  void forwardPass(const MatrixXf& input);
  float getHingeLoss(const VectorXi& labels);
  void backwardPass(const MatrixXf& input,
                    const VectorXi& labels);
  E::VectorXf copyOutput(){return output;}

  void runEpoch();
  void testOnSet(const SampleMatrix& set,
                 float& accuracy,
                 float& hinge_loss);

  void store();
  void load();
};


#endif
