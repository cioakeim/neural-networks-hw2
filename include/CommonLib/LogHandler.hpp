#ifndef LOG_HANDLER_HPP
#define LOG_HANDLER_HPP

#include <fstream>
#include <string>
#include <chrono>


/**
 * @brief Handles timing and recording of different 
 * metrics using csv files.
*/
class LogHandler{
private:
  std::string log_filename; //< Name of log (csv format)
  std::ofstream log_stream; //< Stream of log
  std::chrono::high_resolution_clock::time_point start; //< Time start
  std::chrono::high_resolution_clock::time_point end; //< Time end
  double time_sec; //< Time interval in seconds

public:
  // Empty Constructor 
  LogHandler();
  // Constructor
  LogHandler(std::string log_filename);
  // Destructor 
  ~LogHandler();
  
  // Starts a timer
  void start_timer();
  // Stops the timer
  void end_timer();
  // Time interval to double (seconds)
  double elapsed_seconds();
  // Adds a header to the csv file 
  void add_log_header(std::string header);
  // Logs the time inteval as well as an int denoting size 
  void log_time_and_size(int size);
  // Log the time, the int and the value 
  void log_time_size_value(int size,double value);
};

#endif // !LOG_HANDLER_HPP
