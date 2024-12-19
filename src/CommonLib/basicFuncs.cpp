#include "CommonLib/basicFuncs.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

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


SampleMatrix extract1v1Dataset(const SampleMatrix& full_dataset,
                               int class_1_id,
                               int class_2_id){
  SampleMatrix result;
  return result;  
}


SampleMatrix extract1vAllDataset(const SampleMatrix& full_dataset,
                                 int class_id);
