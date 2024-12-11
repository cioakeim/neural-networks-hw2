#include "MLP/Modifiers.hpp"
#include <iostream>

#define EPSILON 1e-8

Dropout::Dropout():
  rate(0){};

Dropout::Dropout(MatrixXf input, const float rate):
  rate(rate){
  gen=std::mt19937(42);
  dist=std::uniform_real_distribution<float>(0,1);
  mask=MatrixXf(input.rows(),input.cols());
}

void Dropout::maskInput(E::MatrixXf& input){
  // Generate mask 
  mask=(Eigen::MatrixXf::NullaryExpr(input.rows(),input.cols(), [&]() {
      return dist(gen) > rate ? 1.0f : 0.0f;
  }));
  // Apply
  input=input.array()*mask.array();
}

Adam::Adam(float rate,float beta_1,float beta_2,
           MatrixXf& weights,VectorXf& biases,int batch_size):
  epsilon(EPSILON),
  rate(rate/batch_size),beta_1(beta_1),beta_2(beta_2),
  beta_1t(beta_1),beta_2t(beta_2){
  this->m_w=MatrixXf(weights.rows(),weights.cols()).setZero();
  this->u_w=MatrixXf(weights.rows(),weights.cols()).setZero();
  this->m_b=VectorXf(biases.size()).setZero();
  this->u_b=VectorXf(biases.size()).setZero();
}

void Adam::update(const MatrixXf& weight_gradients,
                  const VectorXf& bias_gradients,
                  MatrixXf& weights,
                  VectorXf& biases){
  // Moment calculation
  m_w=beta_1*m_w+(1-beta_1)*weight_gradients;
  u_w.array()=beta_2*u_w.array()+
              (1-beta_2)*weight_gradients.array().square();

  m_b=beta_1*m_b+(1-beta_1)*bias_gradients;
  u_b.array()=beta_2*u_b.array()+
              (1-beta_2)*bias_gradients.array().square();

  // Calculate weights
  weights.array()-=(rate/(1-beta_1t))*
    (m_w.array()/((u_w.array()/(1-beta_2t)).sqrt()+epsilon));
  biases.array()-=(rate/(1-beta_1t))*
    (m_b.array()/((u_b.array()/(1-beta_2t)).sqrt()+epsilon));

  // Diminish beta_t
  beta_1t*=beta_1t;
  beta_2t*=beta_2t;

}
