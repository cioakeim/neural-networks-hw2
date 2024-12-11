#ifndef NEW_LAYER_HPP
#define NEW_LAYER_HPP 

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "MLP/Modifiers.hpp"


#define WEIGHT_DECAY 1e-7

namespace E=Eigen;
using E::MatrixXf;
using E::VectorXf;
using E::VectorXi;
using VectorFunction = std::function<MatrixXf(const MatrixXf)>;

struct Layer{
  MatrixXf weights;
  VectorXf biases;
  MatrixXf activations;
  MatrixXf errors;
  // Adam for each layer
  Adam adam; 

  Layer(int input_size,int output_size,int batch_size):
    weights(MatrixXf(output_size,input_size)),
    biases(VectorXf(output_size)),
    activations(MatrixXf(output_size,batch_size)),
    errors(MatrixXf(output_size,batch_size)){
  };

  Layer(std::string folder_path,const int batch_size);

  void setAdam(float rate,float beta_1,float beta_2,int batch_size){
    this->adam=Adam(rate,beta_1,beta_2,this->weights,this->biases,batch_size);
  }

  // Simple prints
  void printWeights();
  void printBiases();
  void printActivations();
  void printErrors();

  // Method with and without dropout
  void activate(const MatrixXf& input,VectorFunction activation_function,
                Dropout& drop);
  void activate(const MatrixXf& input,VectorFunction activation_function);
  // Softmax output
  void softMaxForward(const MatrixXf& input,VectorFunction activation_function);
  void softMaxBackward(const VectorXi& correct_labels);
  // Back propagation
  void activateErrors(const MatrixXf& next_weights,
                      const MatrixXf& next_errors,
                      VectorFunction activation_derivative);


  // Standard update
  void updateWeights(const MatrixXf& input,
                     const float rate, 
                     const int batch_size);

  // Store to location (2 files, 1 for weights and 1 for bias)
  void store(std::string folder_path);
  // Initializes the weights to random small values.
  void assertRandomWeights();
  // He Initialization accoutning for fan-in 
  void HeRandomInit();
};

#endif
