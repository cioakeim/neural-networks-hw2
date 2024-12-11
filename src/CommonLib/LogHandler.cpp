#include "CommonLib/LogHandler.hpp"
#include <iostream>

// Empty 
LogHandler::LogHandler(){};

// Constructor
LogHandler::LogHandler(std::string log_filename)
  : log_filename(log_filename){
  this->log_stream.open(this->log_filename,std::ios::app);
  if(!this->log_stream.is_open()){
    std::cerr<< "Error in opening log file: "<<this->log_filename<<std::endl;
    exit(1);
  }
}


// Destructor
LogHandler::~LogHandler(){
  this->log_stream.close();
}


// Start time
void LogHandler::start_timer(){
  this->start=std::chrono::high_resolution_clock::now();
}


// End time
void LogHandler::end_timer(){
  this->end=std::chrono::high_resolution_clock::now();
}


// Time to seconds
double LogHandler::elapsed_seconds(){
  this->time_sec=std::chrono::duration<double>(this->end-this->start).count();
  return this->time_sec;
}


// Adds a header to the csv file 
void LogHandler::add_log_header(std::string header){
  this->log_stream<<header<<std::endl;
}

// Logs the time inteval as well as a double on the log 
void LogHandler::log_time_and_size(int size){
  this->log_stream<<
    size<<","<<this->elapsed_seconds()<<std::endl;
}

// Log the time, the int and the value 
void LogHandler::log_time_size_value(int size,double value){
  this->log_stream<<
    size<<","<<this->elapsed_seconds()<<","<<value<<std::endl;
}
