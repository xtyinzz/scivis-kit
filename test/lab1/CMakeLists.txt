cmake_minimum_required(VERSION 3.2)
project(lab1)

# set exec path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# message(-status, ${PROJECT_SOURCE_DIR}, ${SRC_DIR}, ${INCLUDE_DIR}, ${PROJECT_NAME})

file(GLOB SOURCE_FILES "*.cpp")


include_directories("$ENV{NETCDF}/include")
link_libraries("-L$ENV{NETCDF}/lib")

# include_directories($ENV{NETCDF_CFLAGS})
# get executable files
foreach (test_file ${SOURCE_FILES})
  get_filename_component(fname ${test_file} NAME_WE)
  # add_compile_options( $ENV{NETCDF_CFLAGS} $ENV{NETCDF_LIBS} )

  add_executable(${fname} ${test_file})
  target_link_libraries(${fname} netcdf_c++4)
  target_link_libraries(${fname} netcdf)
  # target_link_libraries(${fname} ${NetCDF})
  # target_include_directories(${fname} PUBLIC $ENV{NETCDF_CFLAGs})
  # target_link_libraries(${fname} $ENV{NETCDF_LIBS})
  # target_include_directories(${fname} PUBLIC $ENV{NETCDF_CFLAGs})
  # target_link_libraries(${fname} $ENV{NETCDF_LIBS})
  target_link_libraries(${fname} scivis)
endforeach (test_file)