#ifndef EVENT_TIMER_HPP
#define EVENT_TIMER_HPP 

#include <chrono>
#include <vector>
#include <string>
#include <iostream>

struct Event{
  std::string label;
  double time;
};

class EventTimer{
private:
  bool isRunning = false;
  std::string currentEventLabel;
  std::chrono::high_resolution_clock::time_point startTime;
  std::vector<Event> events;

public:
  void start(std::string label);
  void stop();

  void displayIntervals();

};

#endif
