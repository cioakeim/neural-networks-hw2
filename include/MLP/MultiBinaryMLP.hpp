#ifndef MULTIBINARY_MLP_HPP
#define MULTIBINARY_MLP_HPP 

#include "MLP/SimpleMLP.hpp"


#define CLASS_NUMBER 10
#define MLP_NUMBER (CLASS_NUMBER*(CLASS_NUMBER-1)/2)

/**
 * @brief Combination of multiple 1v1 MLPs for classification
*/
class MBMLP{
private:
  // Filepath
  std::string filepath;

  // Dataset
  const SampleMatrix& training_set;
  const SampleMatrix& test_set;
  std::vector<E::MatrixXf> train_class_sets;
  std::vector<E::MatrixXf> test_class_sets;

  // Mapping
  int pair_to_mlp[CLASS_NUMBER][CLASS_NUMBER];
  SimpleMLP* mlps[MLP_NUMBER];

public:
  MBMLP(const SampleMatrix& training_set,
        const SampleMatrix& test_set,
        const float learning_rate,
        int input_size,
        int batch_size);

  ~MBMLP();

  void setPath(std::string filepath);

  SimpleMLP* getMLPPointer(int class_1_idx,int class_2_idx){
    return mlps[pair_to_mlp[class_1_idx][class_2_idx]];
  }

  const SampleMatrix& getTrainingSetRef(){return this->training_set;}
  const SampleMatrix& getTestSetRef(){return this->test_set;}

  void storeMLP();
  void loadMLP();

  void randomInit();
  void loadDatasets();
  void runEpochBatch(int epoch_batch);

  float trainHingeLoss();
  float testHingeLoss();

  void testOnSet(const SampleMatrix& set,
                 float& accuracy,
                 float& hinge_loss);


  
};

#endif
