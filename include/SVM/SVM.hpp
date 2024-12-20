#ifndef SVM_HPP
#define SVM_HPP

#include <functional>
#include <osqp/osqp.h>
#include "CommonLib/basicStructs.hpp"
#include "CommonLib/eventTimer.hpp"
#include "SVM/Kernels.hpp"

/**
 * @brief A from-scratch SVM implementation, 1v1 class
*/
class SVM{
private:
  // References to the 2 classes
  const E::MatrixXf& class_1_train;
  const E::MatrixXf& class_1_test;
  const E::MatrixXf& class_2_train;
  const E::MatrixXf& class_2_test;
  // Dataset (constructed from the above when training is needed)
  SampleMatrix training_set;
  SampleMatrix test_set;

  // SVM Settings 
  std::function<E::MatrixXf(const E::MatrixXf&,const E::MatrixXf&,KernelParameters)> kernel; //< K(x_i,x_j)
  KernelParameters kernel_parameters;
  float C; //< Parameter for slack variables
  const float lagrange_pruning_threshold=1e-3; //< For finding non-zero lagrange scalars
  const float kernel_matrix_pruning_threshold=1e-6; //< For pruning kernel elements

  // Solver variables
  OSQPSolver* solver;
  // Cost function data
  OSQPCscMatrix P; //< Upper-triangular of Quadratic matrix
  OSQPFloat* q; //< Linear term of cost
  // Constraints (l<Ax<u)
  OSQPCscMatrix A; //< Contraint matrix
  OSQPFloat* l; //< Lower bound
  OSQPFloat* u; //< Upper bound
  // Dimensions of problem
  const OSQPInt m; //< Number of contraints
  const OSQPInt n; //< Number of variables
  // Solver settings 
  OSQPSettings settings;
  
  // The results of training
  E::VectorXf lagrange_times_labels; //< Each element is a_i*y_i for usage.
  E::MatrixXf support_vectors; // Copied to achieve high testing throughput
  float b; // Computed using avg

  // Event timer for keeping track of time intervals
  EventTimer et;

  // Location of svm's store space 
  std::string folder_path;


public:
  // Constructors
  // Get dataset ready for use
  SVM(SampleMatrix& training_set,SampleMatrix& test_set);
  // Get only References to 2 class sets and build train set when needed
  SVM(const E::MatrixXf& class_1_train,
      const E::MatrixXf& class_1_test,
      const E::MatrixXf& class_2_train,
      const E::MatrixXf& class_2_test);

  ~SVM();

  // Setters/getters (only ones needed are implemented)

  // For configuring
  void setKernelFunction(std::function<E::MatrixXf(const E::MatrixXf&,
                                                   const E::MatrixXf&,
                                                   KernelParameters)>func,
                         KernelParameters kernel_parameters){
    this->kernel=func;
    this->kernel_parameters=kernel_parameters;
  }
  void setC(float C){this->C=C;}
  void setFolderPath(std::string folder_path){this->folder_path=folder_path;}


  const SampleMatrix& getTrainingSetRef(){return training_set;}
  const SampleMatrix& getTestSetRef(){return test_set;}

  // Store the solution
  void storeToFile();
  // Load solution
  void loadFromFile();

  // Workflow

  // (Optional) Construct and free the complete dataset
  void constructDatasetFromClassSets();
  void clearDataset();
  void clearSolution();

  // Compute the Kernel matrix for the opt problem
  void computeKernelMatrix();

  void configLinearCostTerm();

  void configConstraints();

  void solveQuadraticProblem();

  void storeSupportVectors();

  // Usage of model

  E::VectorXf output(const E::MatrixXf& samples);

  E::VectorXi predictSet(const E::VectorXf& output);

  void testOnSet(const SampleMatrix& set,
                 float& accuracy,
                 float& mean_hinge_loss); 

  void testOnTrainingSet(float& accuracy,
                         float& mean_hinge_loss);

  void testOnTestSet(float& accuracy,
                     float& mean_hinge_loss);

  // Wrappers

  void solveAndStore();


  // For event timer
  void displayCurrentIntervals(){et.displayIntervals();}

  void storeEventsToFile();
  void startEvent(std::string label){et.start(label);}
  void stopEvent(){et.stop();}

  // Safety checks
  bool areSupportVectorsEmpty(){return support_vectors.cols()<=0;}

};


#endif // !SVM_HPP
