#ifndef BASIC_FUNCS_HPP
#define BASIC_FUNCS_HPP 

#include <string>
#include <filesystem>

namespace fs=std::filesystem;

void ensure_a_path_exists(std::string file_path);

std::string create_network_folder(std::string folder_path);

int count_directories_in_path(const fs::path& path);



#endif
