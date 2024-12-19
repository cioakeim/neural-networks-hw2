#ifndef BASIC_STRUCTS_HPP
#define BASIC_STRUCTS_HPP

#include <Eigen/Dense>

namespace E=Eigen;

// A sample is a N-D point with a class label
struct SamplePoint{
  E::VectorXf vector; //< The sample in Nd 
  uint8_t label; //< The sample's label
};

// A collection of samples packed in a matrix (columns are samples)
struct SampleMatrix{
  E::MatrixXf vectors;
  E::VectorXi labels;
};


#endif // !BASIC_STRUCTS_HPP
