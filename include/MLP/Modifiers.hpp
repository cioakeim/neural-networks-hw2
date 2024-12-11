#ifndef MODIFIERS_HPP
#define MODIFIERS_HPP

#include <random>
#include <Eigen/Dense>

namespace E=Eigen;
using E::MatrixXf;
using E::VectorXf;
using E::VectorXi;

// For dropout creation
struct Dropout{
  const float rate;
  std::mt19937 gen;
  std::uniform_real_distribution<float> dist;
  Eigen::MatrixXf mask;

  Dropout();

  Dropout(MatrixXf input, const float rate);

  void maskInput(E::MatrixXf& input);
};


struct Adam{
  float epsilon;
  float rate;
  float beta_1,beta_2;
  int beta_1t,beta_2t;
  // For weight momentum
  MatrixXf m_w;
  MatrixXf u_w;
  // For bias momentum
  MatrixXf m_b;
  MatrixXf u_b;

  Adam():
    epsilon(0),
    rate(0),
    beta_1(0),beta_2(0){};

  Adam(float rate,float beta_1,float beta_2,
       MatrixXf& weights,VectorXf& biases,int batch_size);

  void update(const MatrixXf& weight_gradients,
              const VectorXf& bias_gradients,
              MatrixXf& weights,
              VectorXf& biases);


};


#endif
