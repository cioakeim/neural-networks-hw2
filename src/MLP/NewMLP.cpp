#include "MLP/NewMLP.hpp"
#include "CommonLib/basicFuncs.hpp"
#include <fstream> 
#include <filesystem>
#include <string>
#include <iostream>
#include <random>

namespace fs=std::filesystem;

// Initialize with trash
MLP::MLP(const std::vector<int>& layer_sizes,
    const int input_size,
    VectorFunction activation_function,
    VectorFunction activation_derivative,
    float learning_rate,
    int batch_size):
  depth(layer_sizes.size()),
  activation_function(std::move(activation_function)),
  activation_derivative(std::move(activation_derivative)),
  learning_rate(learning_rate),
  batch_size(batch_size){
  const float rate=0.2;
  layers.emplace_back(input_size,layer_sizes[0],batch_size);
  drop.emplace_back(layers[0].activations,rate);
  for(int i=1;i<layer_sizes.size();i++){
    layers.emplace_back(layer_sizes[i-1],layer_sizes[i],
                        batch_size);
    drop.emplace_back(layers[i].activations,rate);
  }
   
}


// Load from file
MLP::MLP(std::string file_path,std::string name,
         VectorFunction activation_function,
         VectorFunction activation_derivative,
         float learning_rate,int batch_size):
  activation_function(std::move(activation_function)),
  activation_derivative(std::move(activation_derivative)),
  learning_rate(learning_rate),
  batch_size(batch_size){
  fs::path path(file_path+"/"+name);
  int layer_count=count_directories_in_path(path);
  for(int i=0;i<layer_count;i++){
    layers.emplace_back(file_path+"/"+name+"/layer_"+std::to_string(i),
                        batch_size);
  }
  this->depth=layer_count;
}


void MLP::activateAdam(float beta_1,float beta_2){
  for(int i=0;i<depth;i++){
    layers[i].setAdam(learning_rate, beta_1, beta_2, batch_size);
  }
}

// Do forward and backward pass in batches
void MLP::forwardBatchPass(const MatrixXf& input){
  // Initial layer
  layers[0].activate(input,activation_function,drop[0]);

  for(int i=1;i<depth-1;i++){
    layers[i].activate(layers[i-1].activations,activation_function,drop[i]);
  }
  // Softmax output 
  layers[depth-1].softMaxForward(layers[depth-2].activations,activation_function);
}

float MLP::getBatchLosss(const VectorXi& correct_labels){
  const float epsilon=1e-7;
  float sum=0;

  const int sample_count=correct_labels.size();
  for(int i=0;i<sample_count;i++){
    sum-=log(layers[depth-1].activations(correct_labels[i],i)+epsilon); 
  }
  return sum/sample_count;
}

void MLP::backwardBatchPass(const MatrixXf& input,
                       const VectorXi& correct_labels){
  // Initial errors
  layers[depth-1].softMaxBackward(correct_labels);
  // Backward propagate 
  for(int i=depth-2;i>=0;i--){
    layers[i].activateErrors(layers[i+1].weights, 
                             layers[i+1].errors, 
                             activation_derivative);
  }  
  // Reduce errors and update
  layers[0].updateWeights(input,learning_rate,batch_size);
  #pragma omp parallel for
  for(int i=1;i<depth;i++){
    layers[i].updateWeights(layers[i-1].activations,learning_rate,batch_size);
  }
}

void MLP::shuffleDataset(){
  int training_size=training_set.cols();
  // Shuffle training set 
   // Generate a random permutation of column indices
  std::vector<int> indices(training_size);
  std::iota(indices.begin(), indices.end(), 0);  // Fill indices with 0, 1, ..., cols-1
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(indices.begin(), indices.end(), gen);
  for(int i=0;i<training_size;i++){
    training_set.col(i).swap(training_set.col(indices[i]));
    int temp=training_labels[i];
    training_labels[i]=training_labels(indices[i]);
    training_labels[indices[i]]=temp;
  }

}

float MLP::runEpoch(){
  int training_size=training_set.cols();
  shuffleDataset();
  VectorXf batch_losses=VectorXf(training_size/batch_size);

  for(int idx=0;idx<training_size;idx+=batch_size){
    const MatrixXf& input=training_set.middleCols(idx,batch_size);
    const VectorXi& labels=training_labels.segment(idx,batch_size);
    forwardBatchPass(input);
    batch_losses[idx/batch_size]=getBatchLosss(labels);
    backwardBatchPass(input,labels);
  }
  return batch_losses.mean();
}


void MLP::forwardPassNoDropout(const MatrixXf& input){
  // Initial layer
  layers[0].activate(input,activation_function);

  for(int i=1;i<depth-1;i++){
    layers[i].activate(layers[i-1].activations,activation_function);
  }
  // Softmax output 
  layers[depth-1].softMaxForward(layers[depth-2].activations,activation_function);
}

void MLP::testModel(float& J_test,float& accuracy){
  const int batch_size=(1000<test_labels.size())?(1000):(test_labels.size());
  std::cout<<"Batch size"<<batch_size<<std::endl;
  const int test_size=test_set.cols();

  // Counters of both J_test and accuracy
  int success_count=0; 
  VectorXf batch_losses=VectorXf(test_size/batch_size);
  // Test in batches
  for(int idx=0;idx<test_size;idx+=batch_size){

    // Get columns needed
    const MatrixXf& input=test_set.middleCols(idx,batch_size);
    const VectorXi& labels=test_labels.segment(idx,batch_size);
    // Feed forward with no dropout
    forwardPassNoDropout(input);
    // Record batch loss
    batch_losses[idx/batch_size]=getBatchLosss(labels);

    // Count successful predictions
    for(int i=0;i<batch_size;i++){
      E::Index c_idx;
      layers[depth-1].activations.col(i).maxCoeff(&c_idx);
      if(c_idx==labels[i]){
        success_count++;
      }
    }
  }
  J_test=batch_losses.mean();
  accuracy=static_cast<float>(success_count)/test_size;
}

// I/O
void MLP::store(){
  std::cout<<"Storing network"<<std::endl;
  // Create directory
  std::cout<<store_path<<std::endl;
  fs::path dir(store_path);
  std::cout<<dir.string()<<std::endl;
  if(!fs::exists(dir)){
    fs::create_directories(dir);
  }
  std::ofstream os;
  for(int i=0;i<layers.size();i++){
    // Open main module
    std::string folder=dir.string()+"/layer_"+std::to_string(i);
    fs::create_directory(folder);
    // Store files
    layers[i].store(folder);
  }
}



// Config:
void MLP::randomInit(){
  for(int i=0;i<depth;i++){
    layers[i].HeRandomInit();
  }
}


void MLP::insertDataset(std::vector<SamplePoint>& training_set,
                        std::vector<SamplePoint>& test_set){
  this->training_set=MatrixXf(training_set[0].vector.size(),training_set.size());
  this->training_labels=VectorXi(training_set.size());
  this->test_set=MatrixXf(test_set[0].vector.size(),test_set.size());
  this->test_labels=VectorXi(test_set.size());
  for(int i=0;i<this->training_set.cols();i++){
    this->training_set.col(i)=training_set[i].vector; 
    this->training_labels[i]=training_set[i].label;
  }
  for(int i=0;i<this->test_set.cols();i++){
    this->test_set.col(i)=test_set[i].vector; 
    this->test_labels[i]=test_set[i].label;
  }
}









