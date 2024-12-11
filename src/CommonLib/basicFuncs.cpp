#include "CommonLib/basicFuncs.hpp"

#include <filesystem>

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
