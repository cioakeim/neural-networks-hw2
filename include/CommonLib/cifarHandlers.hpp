#ifndef CIFAR_HANDLERS_HPP
#define CIFAR_HANDLERS_HPP

#include <string>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include "basicStructs.hpp"


/**
 * @brief Manages the input and interpretation of 
 * the CIFAR 10 dataset.
*/
class Cifar10Handler{
private:
  // Vector handling
  const int space_dimension=32*32*3; //< Dimension of each sample in the dataset.
  // File handling
  std::string dataset_folder_path; //< Where the binary files are.
  std::ifstream batch_file_streams[5]; //< Stream handlers for each batch file.
  std::ifstream test_batch_stream; //< Test batch.
  std::string id_to_class_name_lut[10]; //< From class_id to class name.

public:
  // Only logical constructor
  Cifar10Handler(std::string path); 

  // Destructor
  ~Cifar10Handler();

  /**
   * @brief Get an entry from a specific batch 
   *
   * @param[in] batch_id The number of the batch chosen (0 to 4)
   * Set to -1 for test data
   * @param[out] output The N-D point representing the entry (with class_id)
   *
   * @return 0 on success. -1 on EOF.
  */
  int getBatchEntry(int batch_id,SamplePoint& output);


  /**
   * @brief Gets the first N samples from the training files.
   *
   * Returns a vector with the amount of samples that were gathered.
   *
   * @param[in] count Number of samples to be returned.
   *
   * @return Vector of returned samples.
   */
  std::vector<SamplePoint> getTrainingList(int count);


  // Same as above with test samples
  std::vector<SamplePoint> getTestList(int count);


  /**
   * @brief Maps class_id to name.
   *
   * @param[in] class_id ID of the class.
   *
   * @return String of the class
  */
  std::string getClassName(int class_id);


  /**
   * @brief Shows the image version of the sample. 
   *
   * Press q to quit.
   *
   * @param[in] sample The sample to be shown.
  */
  //void printSample(SamplePoint& sample);

};



#endif 
