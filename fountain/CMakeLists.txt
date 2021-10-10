cmake_minimum_required(VERSION 3.0.2)
project(imcv)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g3 -O0")# -g3

find_package(OpenCV 3.4.11 REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(main
  f_load.cpp
)

target_link_libraries(main
  ${OpenCV_LIBRARIES}
)

# for debug
message(WARNING "prefix ${CMAKE_PREFIX_PATH}")    

message("-- OpenCV --")
message(WARNING "version ${OpenCV_VERSION}")
message(WARNING "install path ${OpenCV_INSTALL_PATH}") 
message(WARNING "config path ${OpenCV_CONFIG_PATH}") # look at the output of this message
message(WARNING "libs ${OpenCV_LIBS}")
message(WARNING "include dirs ${OpenCV_INCLUDE_DIRS}")

