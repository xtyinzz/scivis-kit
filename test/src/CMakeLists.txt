cmake_minimum_required(VERSION 3.2)
project(lab2)

# set exec path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# message(-status, ${PROJECT_SOURCE_DIR}, ${SRC_DIR}, ${INCLUDE_DIR})

file(GLOB SOURCE_FILES "*.cpp")

# get executable files
foreach (test_file ${SOURCE_FILES})
  get_filename_component(fname ${test_file} NAME_WE)
  add_executable(${fname} ${test_file})
  target_link_libraries(${fname} scivis)
endforeach (test_file)