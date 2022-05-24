cmake_minimum_required(VERSION 3.2)
project(lab3)

# set exec path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# message(-status, ${PROJECT_SOURCE_DIR}, ${SRC_DIR}, ${INCLUDE_DIR})

file(GLOB SOURCE_FILES "torchTest.cpp")
set(SOURCE_FILES ${SOURCE_FILES} "render_neural.cpp" "render.cpp")


find_package(Torch REQUIRED HINTS "$ENV{HOME}/libs/libtorch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# get executable files
foreach (test_file ${SOURCE_FILES})
  get_filename_component(fname ${test_file} NAME_WE)
  add_executable(${fname} ${test_file})
  target_link_libraries(${fname} scivis)
  target_link_libraries(${fname} ${TORCH_LIBRARIES})
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
  target_include_directories(${fname} INTERFACE ${TORCH_INCLUDE_DIRS})
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
endforeach (test_file)