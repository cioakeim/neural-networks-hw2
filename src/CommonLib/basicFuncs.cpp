#include "CommonLib/basicFuncs.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
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


SampleMatrix extract1vAllDataset(const SampleMatrix& full_dataset,
                                 int class_id);
