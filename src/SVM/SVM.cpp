#include "SVM/SVM.hpp"
#include <osqp/osqp_api_functions.h>
#include <osqp/osqp_api_utils.h>
#include <Eigen/Sparse>
#include <iostream>
#include <algorithm>
#include "CommonLib/basicFuncs.hpp"
#include "omp.h"
#include "CommonLib/eventTimer.hpp"


static void setOSQPMatrixToNull(OSQPCscMatrix& mat){
  mat.x=nullptr;
  mat.i=nullptr;
  mat.p=nullptr;
  mat.n=mat.m=0;
  mat.nzmax=0;
  mat.nz=-1;
}


static void freeOSQPMatrix(OSQPCscMatrix& mat){
  if(mat.x!=nullptr){
    delete[] mat.x;
  }
  if(mat.i!=nullptr){
    delete[] mat.i;
  }
  if(mat.p!=nullptr){
    delete[] mat.p;
  }
}


SVM::SVM(SampleMatrix& training_set,SampleMatrix& test_set):
  class_1_test(training_set.vectors),
  class_2_test(test_set.vectors),
  class_1_train(training_set.vectors),
  class_2_train(training_set.vectors),
  m(1+training_set.vectors.cols()),
  n(training_set.vectors.cols()){
  // Set all matrix pointers to 0
  this->training_set=training_set;
  this->test_set=test_set;
  solver=nullptr;
  setOSQPMatrixToNull(P);
  q=nullptr;
  setOSQPMatrixToNull(A);
  l=u=nullptr;

};


SVM::SVM(const E::MatrixXf& class_1_train,
         const E::MatrixXf& class_1_test,
         const E::MatrixXf& class_2_train,
         const E::MatrixXf& class_2_test):
  class_1_train(class_1_train),
  class_1_test(class_1_test),
  class_2_train(class_2_train),
  class_2_test(class_2_test),
  m(1+class_1_train.cols()+class_2_train.cols()),
  n(class_1_train.cols()+class_2_train.cols()){
  // Init pointers to null for safety.
  solver=nullptr;
  setOSQPMatrixToNull(P);
  q=nullptr;
  setOSQPMatrixToNull(A);
  l=u=nullptr;
}


SVM::~SVM(){
  freeOSQPMatrix(P);
  if(q!=nullptr){
    delete[] q;
  }
  freeOSQPMatrix(A);
  if(l!=nullptr){
    delete[] l;
  }
  if(u!=nullptr){
    delete[] u;
  }
}


static SampleMatrix matricesToSampleMatrix(const E::MatrixXf& class_1,
                                           const E::MatrixXf& class_2){
  SampleMatrix result;
  result.vectors=E::MatrixXf(class_1.rows(),class_1.cols()+class_2.cols());
  result.vectors<<class_1,class_2;
  // +1 if for class 1, -1 is for class 2 
  result.labels=E::VectorXi(class_1.cols()+class_2.cols());
  result.labels<<E::VectorXi::Ones(class_1.cols()),-E::VectorXi::Ones(class_2.cols());

  return result;
}

void SVM::storeToFile(std::string folder_path){
  storeMatrixToFile(folder_path+"/lagrange_times_labels.csv", 
                    lagrange_times_labels);
  storeMatrixToFile(folder_path+"/support_vectors.csv", support_vectors);
  E::MatrixXf temp(1,1);
  temp(0,0)=b;
  storeMatrixToFile(folder_path+"/bias.csv",temp);
}


void SVM::loadFromFile(std::string folder_path){
  lagrange_times_labels=loadMatrixFromFile(folder_path+"/lagrange_times_labels.csv");
  support_vectors=loadMatrixFromFile(folder_path+"/support_vectors.csv");
  E::MatrixXf temp=loadMatrixFromFile(folder_path+"/bias.csv");
  b=temp(0,0);
}


void SVM::constructDatasetFromClassSets(){
  training_set=matricesToSampleMatrix(class_1_train,class_2_train);
  test_set=matricesToSampleMatrix(class_1_test,class_2_test);
}


void SVM::clearDataset(){
  training_set.vectors.resize(0,0);
  training_set.labels.resize(0);
  test_set.vectors.resize(0,0);
  test_set.labels.resize(0);
}

void SVM::clearSolution(){
  lagrange_times_labels.resize(0);
  support_vectors.resize(0,0);

}


static void eigenDenseToOSQPSparse(const E::MatrixXf& eigen,OSQPCscMatrix& osqp){
  E::SparseMatrix<float> sparse=eigen.sparseView();
  // Set matrix dimensions
  osqp.m=sparse.rows();
  osqp.n=sparse.cols();
  // Allocate memory
  osqp.nzmax=sparse.nonZeros();
  osqp.p=new OSQPInt[osqp.n+1];
  osqp.i=new OSQPInt[osqp.nzmax];
  osqp.x=new OSQPFloat[osqp.nzmax];
  // Copy
  std::copy(sparse.outerIndexPtr(),sparse.outerIndexPtr()+osqp.n+1,osqp.p);
  std::copy(sparse.innerIndexPtr(),sparse.innerIndexPtr()+osqp.nzmax,osqp.i);
  std::copy(sparse.valuePtr(),sparse.valuePtr()+osqp.nzmax,osqp.x);
}


void SVM::computeKernelMatrix(){
  // Compute dense kernel.
  //std::cout<<"Calling kernel"<<std::endl;
  const E::MatrixXf denseKernel=kernel(training_set.vectors,
                                       training_set.vectors,
                                       kernel_parameters);
  //std::cout<<"Called kernel"<<std::endl;
  const E::MatrixXf Y=0.5*(training_set.labels*training_set.labels.transpose()).cast<float>();
  E::MatrixXf triang=denseKernel.cwiseProduct(Y.cast<float>()).triangularView<E::Upper>();
  triang=triang.array()/triang.maxCoeff();


  //std::cout<<denseKernel.rows()<<std::endl;
  // Convert to sparse
  const E::SparseMatrix<float> sparse=triang.sparseView();
  //std::cout<<"K: \n"<<sparse<<std::endl;

  eigenDenseToOSQPSparse(triang,P);
}


void SVM::configLinearCostTerm(){
  q=new OSQPFloat[this->n]; 
  const int size=n;
  #pragma omp parallel for
  for(int i=0;i<size;i++){
    q[i]=-1;
  }
}


void SVM::configConstraints(){
  // Compute dense constraint matrix
  E::MatrixXf dense_constraint(n+1,n);
  dense_constraint.row(0)=training_set.labels.cast<float>().transpose();
  dense_constraint.block(1,0,n,n)=E::MatrixXf::Identity(n,n);
  // Convert to Islam
  eigenDenseToOSQPSparse(dense_constraint,A);
  // Lower bound is all zeros
  l=new OSQPFloat[m]();
  u=new OSQPFloat[m];
  u[0]=0;
  std::fill(u+1,u+m,C);
  //std::cout<<dense_constraint<<std::endl;
}


void SVM::solveQuadraticProblem(){
  // Default setup
  osqp_set_default_settings(&settings);
  settings.eps_abs=1e-8;
  settings.eps_rel=1e-8;
  int code;
  code=osqp_setup(&solver,&P,q,&A,l,u,m,n,&settings);
  if(code!=0){
    std::cerr<<"Error.."<<std::endl;
  }
  // Actual solution
  code=osqp_solve(solver);
  if(code!=0){
    std::cerr<<"Error in qsolve.."<<std::endl;
  }


}


void SVM::storeSupportVectors(){
  // Extract all lagrange multipliers
  E::VectorXf a(n);
  std::copy(solver->solution->x,solver->solution->x+n,a.data()); 
  osqp_cleanup(solver);
  //std::cout<<"A: "<<std::endl;
  //std::cout<<a<<std::endl;

  // Filter support vectors (sv) and margin support vectors (msv)
  //
  //std::cout<<"Test"<<std::endl;
  E::ArrayX<bool> a_is_positive = (a.array() > lagrange_pruning_threshold);
  //std::cout<<"Test"<<std::endl;
  E::ArrayX<bool> a_is_less_than_C = (a.array() < (C - lagrange_pruning_threshold));
  //std::cout<<"Test"<<std::endl;
  //E::VectorXi a_is_positive=(a.array()>lagrange_pruning_threshold);
  //E::VectorXi a_is_less_than_C=(a.array()<(C-lagrange_pruning_threshold));
  // Get dimensions of both SV types
  const int sv_nz=a_is_positive.cast<int>().sum();
  //std::cout<<"Test"<<std::endl;
  if(sv_nz==0){
    std::cerr<<"No support vectors found.. Exiting.."<<std::endl;
    exit(1);
  }
  const int msv_nz=(a_is_positive&&a_is_less_than_C).cast<int>().sum();
  /**
  std::cout<<"MSNZ: "<<msv_nz<<std::endl;
  std::cout<<"SNZ: "<<sv_nz<<std::endl;
  std::cout<<"Test"<<std::endl;
  */

  // Allocate memory for stored vectors 
  lagrange_times_labels=E::VectorXf(sv_nz);
  support_vectors=E::MatrixXf(training_set.vectors.rows(),sv_nz);
  // Separate copies for b calculation
  E::MatrixXf msvs=E::MatrixXf(training_set.vectors.rows(),msv_nz);
  E::VectorXf msv_labels=E::VectorXf(msv_nz);
  E::VectorXf msv_a_labels=E::VectorXf(msv_nz);

  int curr_col=0;
  int curr_m_col=0;
  //std::cout<<"Here good\n"<<std::endl;
  for(int i=0;i<sv_nz;i++){
    // If multiplier is posibive
    if(a[i]>lagrange_pruning_threshold){
      // Add to support vectors
      support_vectors.col(curr_col)=training_set.vectors.col(i);
      lagrange_times_labels(curr_col++)=a[i]*training_set.labels[i];
      // If it's also <C 
      if(a[i]<C-lagrange_pruning_threshold){
        msvs.col(curr_m_col)=training_set.vectors.col(i);
        msv_a_labels(curr_m_col)=a[i]*training_set.labels[i];
        msv_labels(curr_m_col++)=training_set.labels[i];
      }
    } 
  }

  const E::MatrixXf x=kernel(msvs,msvs,kernel_parameters)*
                      msv_a_labels;
  b=(msv_labels-x).mean();
  

  /**
  std::cout<<"B: "<<b<<std::endl;
  std::cout<<"Size: "<<x.rows()<<" "<<x.cols()<<std::endl;

  std::cout<<"Labels: "<<training_set.labels.size()<<std::endl;
  */

}


E::VectorXi SVM::predictSet(const E::MatrixXf samples){
  std::cout<<"Sizes: "<<samples.cols()<<" "<<support_vectors.cols()<<" "<<
          lagrange_times_labels.size()<<std::endl;
  const E::VectorXf x=kernel(samples,support_vectors,kernel_parameters)*lagrange_times_labels;
  std::cout<<"End"<<std::endl;
  return (2*((x.array()+b).array()>0).cast<float>()-1).cast<int>();
}


void SVM::testOnSet(const SampleMatrix& set,
                    float& accuracy,
                    float& mean_hinge_loss){
  et.start("Test on set");
  std::cout<<"Check"<<std::endl;
  E::VectorXi pred=predictSet(set.vectors);
  std::cout<<"Check"<<std::endl;

  accuracy=(pred.array()==set.labels.array()).cast<float>().mean();

  std::cout<<"C: "<<C<<" Accuracy: "<<accuracy<<std::endl;
  et.stop();
  
}


void SVM::solveAndStore(){
  std::cout<<"Kernel"<<std::endl;
  std::cout<<"Train size: "<<training_set.vectors.cols()<<" "<<training_set.labels.size()<<std::endl;
  et.start("Compute Kernel Matrix");
  computeKernelMatrix();
  et.stop();
  std::cout<<"Lin"<<std::endl;
  et.start("Configure Linear cost terms");
  configLinearCostTerm();
  et.stop();
  std::cout<<"Constr"<<std::endl;
  et.start("Configure contraints");
  configConstraints();
  et.stop();

  et.start("Solve quadratic problem");
  solveQuadraticProblem();
  et.stop();

  et.start("Store support vectors");
  storeSupportVectors();
  et.stop();

}


void SVM::testOnTrainingSet(float& accuracy,
                            float& mean_hinge_loss){
  testOnSet(training_set,accuracy,mean_hinge_loss);
}

void SVM::testOnTestSet(float& accuracy,
                        float& mean_hinge_loss){
  testOnSet(test_set,accuracy,mean_hinge_loss);
}




