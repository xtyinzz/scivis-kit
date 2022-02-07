cmake_minimum_required(VERSION 2.8)

# Set the project name
project(lab1)
set(SRC_DIR src)
file(GLOB SRC ${SRC_DIR}/task*.cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "-O0 -ggdb")

# get executable files
foreach(srcfile ${SRC})
  get_filename_component(exefile "${srcfile}" NAME_WE)
  add_executable(${exefile} "${PROJECT_SOURCE_DIR}/${SRC_DIR}/${exefile}.cpp")
  target_link_libraries(${exefile} netcdf_c++4)
  target_link_libraries(${exefile} netcdf)
endforeach(srcfile ${SRC})
