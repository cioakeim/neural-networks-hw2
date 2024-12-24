#include "CommonLib/basicFuncs.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <omp.h>

namespace fs=std::filesystem;

void ensure_a_path_exists(std::string file_path){
  fs::path dir(file_path);
  if(!fs::exists(dir)){
    fs::create_directories(dir);
  }
}

std::string create_network_folder(std::string folder_path){
  ensure_a_path_exists(folder_path);
  int current_entry=0;
  while(fs::exists(folder_path+"/network_"+std::to_string(current_entry))){
    current_entry++;
  }
  std::string network_root=folder_path+"/network_"+std::to_string(current_entry);
  fs::create_directory(network_root);
  return network_root;
}

// Auxiliary
int count_directories_in_path(const fs::path& path) {
  int dir_count = 0;
  // Check if the given path exists and is a directory
  if (fs::exists(path) && fs::is_directory(path)) {
    // Iterate through the directory entries
    for (const auto& entry : fs::directory_iterator(path)) {
      // Increment the count for each directory (since we know all entries are directories)
      if(fs::is_directory(entry.path()))
        dir_count++;
    }
  }
  return dir_count;
}


E::MatrixXf loadMatrixFromFile(const std::string file_path){
  std::ifstream file(file_path);
  if(!file.is_open()){
    std::cerr<<"Error in loading: "<<file_path<<std::endl;
    exit(1);
  }
  int rows,cols;
  file>>rows>>cols;
  E::MatrixXf matrix(rows,cols);
  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++){
      file>>matrix(i,j);
    }
  }
  return matrix;
}


void storeMatrixToFile(const std::string file_path,
                       const E::MatrixXf matrix){
  std::ofstream file(file_path);
  if(file.is_open()){
    file<<matrix.rows()<<" "<<matrix.cols()<<"\n"; // Dimensions
    file<<matrix<<"\n";
  }
  else{
    std::cerr<<"Error in storing: "<<file_path<<std::endl;
    exit(1);
  }
}



void shuffleDatasetInPlace(SampleMatrix& set){
  const int training_size=set.vectors.cols();
  // Shuffle training set 
   // Generate a random permutation of column indices
  std::vector<int> indices(training_size);
  std::iota(indices.begin(), indices.end(), 0);  // Fill indices with 0, 1, ..., cols-1
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(indices.begin(), indices.end(), gen);
  for(int i=0;i<training_size;i++){
    set.vectors.col(i).swap(set.vectors.col(indices[i]));
    int temp=set.labels[i];
    set.labels[i]=set.labels(indices[i]);
    set.labels[indices[i]]=temp;
  }
}

void normalizeSet(SampleMatrix& set){
  float sigma=std::sqrt(set.vectors.array().pow(2).mean());
  float mean=set.vectors.array().mean();
  set.vectors.array()-=mean;
  set.vectors.array()/=sigma;
}

std::vector<int> stringToVector(std::string str){
    std::vector<int> result;

    // Use a stringstream to split the input by commas
    std::stringstream ss(str);
    std::string token;
    // Extract integers from the comma-separated string
    while (std::getline(ss, token, ',')) {
        try {
            // Convert each token to an integer and store it in the vector
            result.push_back(std::stoi(token));
        } catch (const std::invalid_argument& e) {
            std::cerr<<"Invalid argument: "<< token <<" is not a valid integer."<< std::endl;
            exit(1);
        }
    }
    return result;
}


std::vector<int> findIndices(const E::VectorXi& labels, int key1, int key2){
  std::vector<int> result;
  for(int i=0;i<labels.size();i++){
    if(labels(i)==key1 || labels(i)==key2){
      result.push_back(i);
    }
  }
  return result;
}


SampleMatrix extract1v1Dataset(const SampleMatrix& full_dataset,
                               int class_1_id,
                               int class_2_id){
  SampleMatrix result;
  std::vector<int> idx_vec=findIndices(full_dataset.labels,class_1_id,class_2_id);
  result.vectors=E::MatrixXf(full_dataset.vectors.rows(),idx_vec.size());
  result.labels=E::VectorXi(idx_vec.size());
  const int size=idx_vec.size();
  #pragma omp parallel for
  for(int i=0;i<size;i++){
    result.vectors.col(i)=full_dataset.vectors.col(idx_vec[i]);
    result.labels(i)=(full_dataset.labels(idx_vec[i])==class_1_id)?+1:-1;
  }
  return result;  
}

static void NaNcheck(const E::MatrixXf& mat,std::string label){
  if(mat.array().isNaN().cast<int>().sum()>0){
    std::cerr<<label<<std::endl;
    exit(1);
  }
}

std::vector<E::MatrixXf> splitDataset(const SampleMatrix& set,int class_number){
  std::vector<std::vector<int>> class_idx(class_number);
  const int sample_size=set.labels.size();
  const int feature_size=set.vectors.rows();
  // Map indices to index vectors
  for(int i=0;i<sample_size;i++){
    class_idx[set.labels(i)].push_back(i);
  }
  // Allocate memory for matrices
  std::vector<E::MatrixXf> result;
  for(int i=0;i<class_number;i++){
    result.emplace_back(feature_size,class_idx[i].size());
  }
  for(int i=0;i<class_number;i++){
    const int class_samples=class_idx[i].size();
    for(int j=0;j<class_samples;j++){
      result[i].col(j)=set.vectors.col(class_idx[i][j]);
    }
  }
  return result;
}


