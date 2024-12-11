#ifndef NEW_MLP_HPP
#define NEW_MLP_HPP

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "MLP/NewLayer.hpp"
#include "MLP/Modifiers.hpp"
#include "CommonLib/basicStructs.hpp"

#define WEIGHT_DECAY 1e-7

namespace E=Eigen;
using E::MatrixXf;
using E::VectorXf;
using E::VectorXi;
using VectorFunction = std::function<MatrixXf(const MatrixXf)>;



class MLP{
protected:
  // For I/O purposes
  std::string store_path;
  // Structure
  std::vector<Layer> layers;
  int depth;
  VectorFunction activation_function;
  VectorFunction activation_derivative;
  // Parameters
  const float learning_rate;
  const int batch_size;
  // Training set
  MatrixXf training_set;
  VectorXi training_labels;
  MatrixXf test_set;
  VectorXi test_labels;

  // Dropout for each layer
  std::vector<Dropout> drop;

  
public:

  MLP(const std::vector<int>& layer_sizes,
      const int input_size,
      VectorFunction activation_function,
      VectorFunction activation_derivative,
      float learning_rate,
      int batch_size);

  MLP(std::string file_path,std::string name,
      VectorFunction activation_function,
      VectorFunction activation_derivative,
      float learning_rate,int batch_size);

  void setStorePath(std::string path){this->store_path=path;}
  void setFunction(VectorFunction f,VectorFunction f_dot){
    this->activation_function=f;
    this->activation_derivative=f_dot;
  }
  void activateAdam(float beta_1,float beta_2);

  // Do forward and backward pass in batches
  void forwardBatchPass(const MatrixXf& input);
  float getBatchLosss(const VectorXi& correct_labels);
  void backwardBatchPass(const MatrixXf& input,
                         const VectorXi& correct_labels);
  // For the whole dataset (assumed the array is shuffled)
  float runEpoch();

  // Running without dropout
  void forwardPassNoDropout(const MatrixXf& input);

  // Test the epoch result (return the loss function and accuracy)
  void testModel(float& J_test,float& accuracy);

  // Store to place
  void store();

  // Config:
  void randomInit();
  void insertDataset(std::vector<SamplePoint>& training_set,
                     std::vector<SamplePoint>& test_set);
  void shuffleDataset();

  
};





#endif
