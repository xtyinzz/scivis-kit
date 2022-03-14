cmake_minimum_required(VERSION 3.2)

# Set the project name
project(scivis)
# set(SRC_DIR src)
# file(GLOB SRC ${SRC_DIR}/task*.cpp)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O1")



# add_library(Boost INTERFACE)
# target_include_directories(Boost INTERFACE "/usr/include")

# add_library(Eigen INTERFACE)
# target_include_directories(Eigen INTERFACE "/usr/include")

# add_library(LINK_LIBS INTERFACE)
# target_link_libraries(LINK_LIBS INTERFACE Boost Eigen)




# add_library(GLM INTERFACE)
# target_include_directories(GLM INTERFACE "/usr/include")

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${INCLUDE_DIR})

set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${SRC_DIR})

# file(GLOB SOURCE_FILES "${SRC_DIR}/*.cpp")
# # file(GLOB SOURCE_HEADERS "${SRC_DIR}/*.h")
# add_library(scivis SHARED ${SOURCE_FILES})
# target_include_directories(scivis INTERFACE ${SRC_DIR})
# link_libraries(scivis)

# add_library(scivis "")

# set(VTK_DIR "$ENV{HOME}/libs/vtk/build")
# message(-status, ${VTK_DIR})
# find_package(VTK REQUIRED)
# include(${VTK_USE_FILE})
# message(-status, ${VTK_DIR})
# message(-status, ${VTK_USE_FILE})
# message(-status, ${VTK_LIBRARIES})
# target_link_libraries(scivis PUBLIC ${VTK_LIBRARIES})


# message(-status, ${PROJECT_NAME})

# include_directories("/usr")
add_subdirectory(${SRC_DIR})
add_subdirectory(test)

