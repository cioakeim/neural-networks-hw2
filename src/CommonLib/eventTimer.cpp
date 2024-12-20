#include "CommonLib/eventTimer.hpp"
#include <fstream>
#include <string>


void EventTimer::start(std::string label){
  if(isRunning==true){
    std::cerr<<"Error in timer, double start.."<<std::endl;
    exit(1);
  }
  this->currentEventLabel=label;
  this->startTime=std::chrono::high_resolution_clock::now();
  isRunning =true;
}

void EventTimer::stop(){
  if(isRunning==false){
    std::cerr<<"Error in timer, stop without start.."<<std::endl;
    exit(1);
  }
  auto endTime=std::chrono::high_resolution_clock::now();
  double duration=std::chrono::duration_cast<std::chrono::duration<double>>(endTime-startTime).count();
  events.push_back({currentEventLabel,duration});

  isRunning =false;
  currentEventLabel.clear();
}


void EventTimer::displayIntervals(){
  for(const auto& event: events){
    std::cout<<event.label<<" (seconds), "<<event.time<<std::endl;
  }
}

void EventTimer::writeToFile(std::string file_path){
  std::ofstream file(file_path);
  if(!file.is_open()){
    std::cerr<<"Error opening: "<<file_path<<std::endl;
    exit(1);
  }
  for(const auto& event: events){
    file<<event.label<<" (s), "<<event.time<<"\n";
  }
  file.close();
}
