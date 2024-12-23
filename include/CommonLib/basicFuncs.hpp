#ifndef BASIC_FUNCS_HPP
#define BASIC_FUNCS_HPP 

#include <string>
#include <filesystem>
#ifndef EIGEN_USE_BLAS
#define EIGEN_USE_BLAS
#endif
#ifndef EIGEN_USE_LAPACK
#define EIGEN_USE_LAPACK
#endif
#include <Eigen/Dense>
#include "CommonLib/basicStructs.hpp"

namespace fs=std::filesystem;
namespace E=Eigen;

void ensure_a_path_exists(std::string file_path);

std::string create_network_folder(std::string folder_path);

int count_directories_in_path(const fs::path& path);

E::MatrixXf loadMatrixFromFile(const std::string file_path);

void storeMatrixToFile(const std::string file_path,
                       const E::MatrixXf matrix);

void normalizeSet(SampleMatrix& set);


std::vector<int> stringToVector(std::string str);


std::vector<int> findIndices(const E::VectorXi& labels, int key1, int key2);


SampleMatrix extract1v1Dataset(const SampleMatrix& full_dataset,
                               int class_1_id,
                               int class_2_id);

SampleMatrix extract1vAllDataset(const SampleMatrix& full_dataset,
                                 int class_id);

std::vector<E::MatrixXf> splitDataset(const SampleMatrix& set,int class_number);


#endif
