#include "CommonLib/eventTimer.hpp"


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
  for(const auto event: events){
    std::cout<<event.label<<": "<<event.time<<" seconds."<<std::endl;
  }
}
