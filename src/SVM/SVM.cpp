#ifndef EIGEN_USE_BLAS
#define EIGEN_USE_BLAS
#endif
#ifndef EIGEN_USE_LAPACK
#define EIGEN_USE_LAPACK
#endif
#include "SVM/SVM.hpp"
#include <osqp/osqp_api_constants.h>
#include <osqp/osqp_api_functions.h>
#include <osqp/osqp_api_utils.h>
#include <Eigen/Sparse>
#include <iostream>
#include <algorithm>
#include "CommonLib/basicFuncs.hpp"
#include "omp.h"
#include "CommonLib/eventTimer.hpp"
#include <cstdlib>


static void setOSQPMatrixToNull(OSQPCscMatrix& mat){
  mat.x=nullptr;
  mat.i=nullptr;
  mat.p=nullptr;
  mat.n=mat.m=0;
  mat.nzmax=0;
  mat.nz=-1;
}


static void freeOSQPMatrix(OSQPCscMatrix* mat){
  if(mat->x!=nullptr){
    free(mat->x);
  }
  if(mat->i!=nullptr){
    free(mat->i);
  }
  if(mat->p!=nullptr){
    free(mat->p);
  }
  free(mat);
}


SVM::SVM(SampleMatrix& training_set,SampleMatrix& test_set):
  class_1_train(training_set.vectors),
  class_1_test(training_set.vectors),
  class_2_train(training_set.vectors),
  class_2_test(test_set.vectors),
  m(1+training_set.vectors.cols()),
  n(training_set.vectors.cols()){
  // Set all matrix pointers to 0
  this->training_set=training_set;
  this->test_set=test_set;
  solver=nullptr;
  P=nullptr;
  q=nullptr;
  A=nullptr;
  l=u=nullptr;

};

static void NaNcheck(const E::MatrixXf& mat,std::string label){
  if(mat.array().isNaN().cast<int>().sum()>0){
    std::cerr<<label<<std::endl;
    exit(1);
  }
}

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
  P=nullptr;
  q=nullptr;
  A=nullptr;
  l=u=nullptr;
}


SVM::~SVM(){
  if(A!=nullptr){
    freeOSQPMatrix(A);
  }
  if(q!=nullptr){
    free(q);
  }
  if(P!=nullptr){
    freeOSQPMatrix(P);
  }
  if(l!=nullptr){
    free(l);
  }
  if(u!=nullptr){
    free(u);
  }
  if(settings!=nullptr){
    free(settings);
  }
}


static SampleMatrix matricesToSampleMatrix(const E::MatrixXf& class_1,
                                           const E::MatrixXf& class_2){
  SampleMatrix result;
  if(class_1.array().isNaN().cast<int>().sum()>0){
    std::cerr<<"Error in class_1!!!"<<std::endl;
    exit(1);
  }
  if(class_2.array().isNaN().sum()>0){
    std::cerr<<"Error in class_2!!!"<<std::endl;
    exit(1);
  }
  result.vectors=E::MatrixXf(class_1.rows(),class_1.cols()+class_2.cols());
  result.vectors<<class_1,class_2;
  if(result.vectors.array().isNaN().sum()>0){
    std::cerr<<"Error in vectors!!!"<<std::endl;
    exit(1);
  }
  // +1 if for class 1, -1 is for class 2 
  result.labels=E::VectorXi(class_1.cols()+class_2.cols());
  result.labels<<E::VectorXi::Ones(class_1.cols()),-E::VectorXi::Ones(class_2.cols());

  return result;
}


void SVM::storeToFile(){
  // Different values for different Cs so: 
  std::string filepath=folder_path+"/C_"+std::to_string(C);
  ensure_a_path_exists(filepath);
  const int sv_sz=sv_indices.size();
  if(sv_sz==0){
    std::cout<<"Empty one.."<<std::endl;
    std::string dummy;
    std::getline(std::cin,dummy);
  }
  E::VectorXf indices(sv_sz);
  for(int i=0;i<sv_sz;i++){
    indices(i)=sv_indices[i];
  }
  storeMatrixToFile(filepath+"/sv_indices.csv",
                    indices);
  std::cout<<"Lagrange dim: "<<lagrange_times_labels.rows()<<" "<<lagrange_times_labels.cols()<<std::endl;
  storeMatrixToFile(filepath+"/lagrange_times_labels.csv", 
                    lagrange_times_labels);
  E::MatrixXf temp(1,1);
  std::cout<<"B: "<<b<<std::endl;
  temp(0,0)=b;
  storeMatrixToFile(filepath+"/bias.csv",temp);
}

void SVM::storeEventsToFile(){
  std::string filepath=folder_path+"/time_info";
  ensure_a_path_exists(filepath);
  et.writeToFile(filepath+"/C_"+std::to_string(C)+".txt");
  et.clearEvents();
}


void SVM::loadFromFile(){
  std::string filepath=folder_path+"/C_"+std::to_string(C);
  E::VectorXf indices=loadMatrixFromFile(filepath+"/sv_indices.csv");
  const int sv_sz=indices.size();
  sv_indices.clear();
  for(int i=0;i<sv_sz;i++){
    sv_indices.push_back(static_cast<int>(indices(i)));
  }
  lagrange_times_labels=loadMatrixFromFile(filepath+"/lagrange_times_labels.csv");
  E::MatrixXf temp=loadMatrixFromFile(filepath+"/bias.csv");
  b=temp(0,0);
}


void SVM::constructDatasetFromClassSets(){
  if(class_1_train.array().isNaN().cast<int>().sum()>0){
    std::cerr<<"Bad class_1_train"<<std::endl;
    exit(1);
  }
  if(class_2_train.array().isNaN().cast<int>().sum()>0){
    std::cerr<<"Bad class_2_train"<<std::endl;
    exit(1);
  }
  training_set=matricesToSampleMatrix(class_1_train,class_2_train);
  if(training_set.vectors.array().isNaN().cast<int>().sum()>0){
    std::cerr<<"Error in training set creation..."<<std::endl;
  }
  test_set=matricesToSampleMatrix(class_1_test,class_2_test);
  if(test_set.vectors.array().isNaN().cast<int>().sum()>0){
    std::cerr<<"Error in test set creation..."<<std::endl;
  }
}


void SVM::clearDataset(){
  training_set.vectors.conservativeResize(0,0);
  training_set.labels.conservativeResize(0);
  test_set.vectors.conservativeResize(0,0);
  test_set.labels.conservativeResize(0);
}


void SVM::clearSupportVectors(){
  support_vectors.conservativeResize(0,0);
}

void SVM::clearWholeSolution(){
  clearSupportVectors();
  lagrange_times_labels.conservativeResize(0);
  sv_indices=std::vector<int>();
  msv_indices=std::vector<int>();

}

void SVM::loadSupportVectors(){
  const int sv_sz=sv_indices.size();
  if(sv_sz==0){
    std::cout<<"ZERO"<<std::endl;
    std::string dummy;
    std::getline(std::cin, dummy);
  }
  support_vectors=E::MatrixXf(class_1_test.rows(),sv_sz);
  #pragma omp parallel for
  for(int i=0;i<sv_sz;i++){
    support_vectors.col(i)=training_set.vectors.col(sv_indices[i]);
  }
}

static void eigenDenseToOSQPSparse(const E::MatrixXd& eigen,OSQPCscMatrix** osqp){
  E::SparseMatrix<double> sparse=eigen.sparseView();
  OSQPInt m=sparse.rows();
  OSQPInt n=sparse.cols();
  // Allocate memory
  OSQPInt nzmax=sparse.nonZeros();
  //std::cout<<"nzmax: "<<nzmax<<std::endl;
  *osqp=(OSQPCscMatrix*)malloc(sizeof(OSQPCscMatrix));
  OSQPInt *p=(OSQPInt*)malloc((n+1)*sizeof(OSQPInt));
  OSQPInt *i=(OSQPInt*)malloc(nzmax*sizeof(OSQPInt));
  OSQPFloat *x=(OSQPFloat*)malloc(nzmax*sizeof(OSQPFloat));
  // Copy
  for(int l=0;l<n+1;l++){
    p[l]=sparse.outerIndexPtr()[l];
  }
  for(int l=0;l<nzmax;l++){
    i[l]=sparse.innerIndexPtr()[l];
    x[l]=sparse.valuePtr()[l];
  }
  csc_set_data(*osqp, m, n, nzmax, x, i, p);
}


void SVM::computeKernelMatrix(){
  // Compute dense kernel.
  //std::cout<<"Calling kernel"<<std::endl;
  et.start("Kernel");
  E::MatrixXd denseKernel=kernel(training_set.vectors,
                                 training_set.vectors,
                                 kernel_parameters).cast<double>();
  et.displayIntervals();
  
  const E::MatrixXd Y=(training_set.labels*training_set.labels.transpose()).cast<double>();


  E::MatrixXd mat=denseKernel.cwiseProduct(Y.cast<double>());


  std::cout<<"Max coeff: "<<mat.maxCoeff()<<std::endl;
  std::cout<<"Min abs coeff: "<<mat.cwiseAbs().minCoeff()<<std::endl;


  E::MatrixXd triang=mat.triangularView<E::Upper>();
  triang=triang.array()/triang.maxCoeff();

  eigenDenseToOSQPSparse(triang,&P);
  et.stop();
}


void SVM::configLinearCostTerm(){
  et.start("Linear");
  q=(OSQPFloat*) malloc(n*sizeof(OSQPFloat)); 
  const int size=n;
  #pragma omp parallel for
  for(int i=0;i<size;i++){
    q[i]=-1;
  }
  et.stop();
}


void SVM::configConstraints(){
  et.start("Constraints");
  // Compute dense constraint matrix
  E::MatrixXd dense_constraint(n+1,n);
  dense_constraint.row(0)=training_set.labels.cast<double>().transpose();
  dense_constraint.block(1,0,n,n)=E::MatrixXd::Identity(n,n);
  // Convert to Islam
  eigenDenseToOSQPSparse(dense_constraint,&A);
  // Lower bound is all zeros
  l=(OSQPFloat*)calloc(m,sizeof(OSQPFloat));
  u=(OSQPFloat*)malloc(m*sizeof(OSQPFloat));
  u[0]=0;
  std::fill(u+1,u+m,C);
  //std::cout<<dense_constraint<<std::endl;
  /**
  for(int i=0;i<n;i++){
    std::cout<<l[i]<<" "<<u[i]<<std::endl;
  }
  */
  et.stop();
}


void SVM::solveQuadraticProblem(){
  et.start("Solve");
  solver=(OSQPSolver*)malloc(sizeof(OSQPSolver));
  settings=(OSQPSettings*)malloc(sizeof(OSQPSettings));
  // Default setup
  osqp_set_default_settings(settings);
  //settings->linsys_solver=OSQP_INDIRECT_SOLVER;
  //settings->polishing=true;
  //settings->linsys_solver=OSQP_DIRECT_SOLVER;
  //
  omp_set_num_threads(8);
  setenv("MKL_DOMAIN_NUM_THREADS", "8", 1);
  //mkl_set_num_threads_local(8);
  //settings->warm_starting=0;
  settings->eps_abs=1e-9;
  settings->eps_rel=1e-9;

  
  
  int code;
  std::cout<<"Setup.."<<std::endl;
  code=osqp_setup(&solver,P,q,A,l,u,m,n,settings);
  if(code!=0){
    std::cerr<<"Error..: "<<osqp_error_message(code)<<std::endl;
  }
  // Actual solution
  std::cout<<"Solving.."<<std::endl;
  code=osqp_solve(solver);
  if(code!=0){
    std::cerr<<"Error in qsolve.."<<std::endl;
  }
  et.stop();
}

void SVM::deconstructQPProblem(){
  freeOSQPMatrix(P);
  freeOSQPMatrix(A);
  free(q);
  free(l);
  free(u);
}


void SVM::storeSVIndicesAndAVector(){
  et.start("Store indices");
  // Limits
  float C_eps=(C<1)?1e-9*C:1e-9;
  float eff_pruning=(C<1)?lagrange_pruning_threshold*C:lagrange_pruning_threshold;
  // Extract all lagrange multipliers
  E::VectorXd a(n);
  std::copy(solver->solution->x,solver->solution->x+n,a.data()); 
  osqp_cleanup(solver);

  // Filter support vectors (sv) and margin support vectors (msv)
  //
  int sv_nz,msv_nz;
  sv_indices.clear();
  msv_indices.clear();
  std::cout<<"Check"<<std::endl;
  for(int i=0;i<100;i++){
    std::cout<<"Try: "<<i<<std::endl;
    // Get candidates for support vectors and margin support vectors
    E::ArrayX<bool> a_is_sv= (a.array() > eff_pruning);
    E::ArrayX<bool> a_is_less_than_C = (a.array() < (C - C_eps));
    E::ArrayX<bool> a_is_msv=(a_is_sv&&a_is_less_than_C);
    sv_nz=a_is_sv.size();
    msv_nz=a_is_msv.size();
    // If no support vectors found, increase tolerance to small values
    // and try again
    if(sv_nz==0){
      eff_pruning/=10;
      continue;
    }
    // Same idea for margin support vectors
    if(msv_nz==0){
      C_eps/=10;
      continue;
    }
    // If both vectors are non-zero, the solution is found
    // Get indices of svs 
    const int size=a.size();
    for(int i=0;i<size;i++){
      if(a_is_sv(i)==true)
        sv_indices.push_back(i);
      if(a_is_msv(i)==true)
        msv_indices.push_back(i);
    }
    sv_nz=sv_indices.size();
    msv_nz=msv_indices.size();
    // And break
    break;
  }
  // MSV and SV indices found
  
  // Save lagrange values 
  lagrange_times_labels=E::VectorXf(sv_nz);
  for(int i=0;i<sv_nz;i++){
    lagrange_times_labels(i)=a[sv_indices[i]]*training_set.labels[sv_indices[i]];
  }

  const int msv_upper_bound=10;
  int msvs_used=(msv_nz<msv_upper_bound)?msv_nz:msv_upper_bound;
  // Calculate b
  E::MatrixXf msvs=E::MatrixXf(training_set.vectors.rows(),msvs_used);
  E::VectorXf msv_labels=E::VectorXf(msvs_used);
  E::VectorXf msv_a_labels=E::VectorXf(msvs_used);
  // MSV map
  for(int i=0;i<msvs_used;i++){
    msvs.col(i)=training_set.vectors.col(msv_indices[i]); 
    msv_labels(i)=training_set.labels[msv_indices[i]];
    msv_a_labels(i)=a[msv_indices[i]]*training_set.labels[msv_indices[i]];
  }

  std::cout<<"MSV NANS: "<<(msvs.array().isNaN().cast<int>().sum())<<std::endl;
  
  const E::MatrixXf x=kernel(msvs,msvs,kernel_parameters)*
                      msv_a_labels;
  std::cout<<"X NANS: "<<(x.array().isNaN().cast<int>()).sum()<<std::endl;
  b=(msv_labels-x).mean();
  std::cout<<"B: "<<b<<std::endl;


  et.stop();
}


void SVM::storeSupportVectors(){
  et.start("Store");
  const int sv_nz=sv_indices.size();
  // Allocate memory for stored vectors 
  support_vectors=E::MatrixXf(training_set.vectors.rows(),sv_nz);
  // SV map
  for(int i=0;i<sv_nz;i++){
    support_vectors.col(i)=training_set.vectors.col(sv_indices[i]); 
  }
  et.stop();
}


E::VectorXf SVM::output(const E::MatrixXf& samples){
  const E::VectorXf wx=kernel(samples,support_vectors,kernel_parameters)*lagrange_times_labels;
  return wx.array()+b;
}


E::VectorXi SVM::predictSet(const E::VectorXf& output){
  return (2*((output.array()).array()>0).cast<float>()-1).cast<int>();
}

float SVM::getHingeLoss(const E::VectorXf& output,
                        const E::VectorXi& labels){
  return (1-(labels.array()).cast<float>()*output.array()).cwiseMax(0).mean();
}

void SVM::testOnSet(const SampleMatrix& set,
                    float& accuracy,
                    float& mean_hinge_loss){
  std::cout<<"Check"<<std::endl;
  std::cout<<"SV DIMS: "<<support_vectors.rows()<<" "<<support_vectors.cols()<<std::endl;
  std::cout<<"SET: "<<set.vectors.rows()<<" "<<set.vectors.cols()<<std::endl;
  // Output
  const E::VectorXf out=output(set.vectors);
  std::cout<<"Check1"<<std::endl;
  // Prediction (+-1)
  const E::VectorXf pred=predictSet(out).cast<float>();

  // Calculate accuracy
  accuracy=(pred.array()==set.labels.cast<float>().array()).cast<float>().mean();
  // For mean hinge loss
  mean_hinge_loss=(1-(set.labels.array()).cast<float>()*out.array()).cwiseMax(0).mean();

  std::cout<<"C: "<<C<<" Accuracy: "<<accuracy<<std::endl;
  std::cout<<"Mean hinge loss: "<<mean_hinge_loss<<std::endl;
}

void SVM::createQPProblem(){
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




