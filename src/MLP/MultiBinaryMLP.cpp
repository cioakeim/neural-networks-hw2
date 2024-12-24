#include "MLP/MultiBinaryMLP.hpp"
#include "CommonLib/basicFuncs.hpp"
#include "CommonLib/eventTimer.hpp"

MBMLP::MBMLP(const SampleMatrix& training_set,
             const SampleMatrix& test_set,
             const float learning_rate,
             int input_size,
             int batch_size):
  training_set(training_set),
  test_set(test_set){
  int x=0;
  for(int class1=0;class1<CLASS_NUMBER-1;class1++){
    for(int class2=class1+1;class2<CLASS_NUMBER;class2++){
      pair_to_mlp[class1][class2]=x++;
    }
  }
  train_class_sets=splitDataset(training_set,CLASS_NUMBER);
  test_class_sets=splitDataset(test_set,CLASS_NUMBER);
  for(int class1=0;class1<CLASS_NUMBER-1;class1++){
    for(int class2=class1+1;class2<CLASS_NUMBER;class2++){
      mlps[pair_to_mlp[class1][class2]]=new SimpleMLP(learning_rate,
                                                      input_size,
                                                      batch_size,
                                                      class1,class2,
                                                      training_set,
                                                      test_set);
    }
  }
}

MBMLP::~MBMLP(){
  for(int i=0;i<MLP_NUMBER;i++){
    delete mlps[i];
  }
}

void MBMLP::setPath(std::string filepath){
  this->filepath=filepath;
  for(int i=0;i<MLP_NUMBER;i++){
    mlps[i]->setStorePath(filepath+"/MLP_"+std::to_string(i));    
  }
}


void storeMLP();
void loadMLP();

void MBMLP::randomInit(){
  for(int i=0;i<MLP_NUMBER;i++){
    mlps[i]->randomInit();
  }
}

void MBMLP::loadDatasets(){
  for(int i=0;i<MLP_NUMBER;i++){
    mlps[i]->loadDataset();
  }
}

void MBMLP::runEpochBatch(int epoch_batch){
  for(int i=0;i<MLP_NUMBER;i++){
    mlps[i]->loadDataset();
    for(int epoch=0;epoch<epoch_batch;epoch++){
      mlps[i]->runEpoch();
    }
    mlps[i]->clearDataset();
  }
  
}


void MBMLP::testOnSet(const SampleMatrix& set,
                      float& accuracy,
                      float& hinge_loss){
  EventTimer my_et;
  int hinge_denominator=0;
  const int sample_size=set.labels.size();
  // For final mean hinge loss
  // In the below matrices, each column is for the prediction of 1 sample
  // Each SVM will place its vote here.
  E::MatrixXf votes=E::MatrixXf::Zero(CLASS_NUMBER,sample_size);
  
  // In the case of a tie, the mean hinge losses are kept here

  my_et.start("Phase1");
  // Each binary SVM votes and places its confidence value in the corresponding bin
  for(int class_1=0;class_1<CLASS_NUMBER-1;class_1++){
    for(int class_2=class_1+1;class_2<CLASS_NUMBER;class_2++){
      SimpleMLP* mlp=mlps[pair_to_mlp[class_1][class_2]];
      // Get output of svm
      mlp->forwardPass(set.vectors);
      const E::VectorXf output=mlp->copyOutput();
      // Get prediction
      const E::VectorXi pred_bin=(output.array()>0).cast<int>();
      // Convert to class ids
      const E::VectorXi pred_class_id=(pred_bin.array()*class_1+(1-pred_bin.array())*class_2);
      // For each prediction place in correct bin
      #pragma omp parallel for
      for(int i=0;i<sample_size;i++){
        votes(pred_class_id(i),i)+=std::abs(output(i));
      }
      // For hinge loss
      const E::VectorXi bin_labels=(set.labels.array()==class_1).array().cast<int>()-
                                   (set.labels.array()==class_2).array().cast<int>();
      hinge_loss+=(1-output.array()*bin_labels.cast<float>().array()).cwiseMax(0).sum();
      if(hinge_loss<0){
        std::cout<<"ERROR"<<std::endl;
      }
      hinge_denominator+=(bin_labels.array()!=0).cast<int>().sum();
    }
  }
  my_et.stop();

  my_et.start("Phase2");
  // Time for voting results
  E::VectorXi final_prediction(sample_size);

  // Reduce column-wise
  #pragma omp parallel for
  for(int col=0;col<sample_size;col++){
    E::VectorXf::Index idx;
    votes.col(col).maxCoeff(&idx);
    final_prediction(col)=idx;
  }
  // Get final prediction
  accuracy=(final_prediction.array()==set.labels.array()).cast<float>().mean();
  hinge_loss=hinge_loss/hinge_denominator;
  my_et.stop();
  my_et.displayIntervals();
}












