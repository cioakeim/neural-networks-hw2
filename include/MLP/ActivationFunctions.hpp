#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include <Eigen/Dense>

namespace E=Eigen;

// Each function comes with its derivative 

// ReLU

E::MatrixXf reLU(const E::MatrixXf& in);
E::MatrixXf reLUder(const E::MatrixXf& reLU_output);

E::MatrixXf reLUdropout(const E::MatrixXf& in);
E::MatrixXf reLUder(const E::MatrixXf& reLU_output);

float reLU_el(const float in);
float reLUder_el(const float in);

// Tanh
E::VectorXf tanh(const E::VectorXf& in);
E::VectorXf tanhder(const E::VectorXf& tanh_output);

#endif
