cmake_minimum_required(VERSION 3.14)
project(registrationGuidedDenoise)

set(CMAKE_CXX_STANDARD 14)

add_executable(registrationGuidedDenoise main.cpp WienerFilter.cpp )

find_package(OpenCV REQUIRED)
target_link_libraries(registrationGuidedDenoise ${OpenCV_LIBS})