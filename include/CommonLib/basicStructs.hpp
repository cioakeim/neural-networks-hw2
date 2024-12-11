#ifndef BASIC_STRUCTS_HPP
#define BASIC_STRUCTS_HPP

#include <Eigen/Dense>

namespace E=Eigen;

// A sample is a N-D point with a class label
struct SamplePoint{
  E::VectorXf vector; //< The sample in Nd 
  uint8_t label; //< The sample's label
};


struct CudaSamplePoint{
  float* vector;
  int size;
  uint8_t label;
};

#endif // !BASIC_STRUCTS_HPP
