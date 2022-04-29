set (SRC_DIRS
  .
  container
  renderer
  common
)

foreach (dir ${SRC_DIRS})
  file(GLOB tmp ${dir}/*.cpp ${dir}/*.cu)
  list(APPEND SRC_FILES ${tmp})
  file(GLOB tmp RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${dir}/*.h)
  list(APPEND HEADER_FILES ${tmp})
endforeach(dir)

# Adding source file and include headers for scivis library
# include_directories(.)
# target_sources(scivis PRIVATE ${SRC_FILES})
add_library(scivis SHARED ${SRC_FILES} ${HEADER_FILES})
set_target_properties(scivis PROPERTIES LINKER_LANGUAGE CXX)

# Link header-only libraries collected root CMakeLists
target_link_libraries(scivis INTERFACE ${LINK_HEADER_LIBS})


find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(scivis PRIVATE OpenMP::OpenMP_CXX)
endif()


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
execute_process (
    COMMAND bash -c "export VTK_DIR=/home/xiong336/libs/vtk/build/"
    OUTPUT_VARIABLE outVar
)
find_package(VTK REQUIRED COMPONENTS
  CommonColor
  CommonCore
  CommonDataModel
  FiltersCore
  IOXML
)
if(VTK_FOUND)
  # message(-status, "111111111111111${VTK_USE_FILE}")
  # message(-status, "111111111111111${VTK_INCLUDE_DIR}")
  # message(-status, "111111111111111${VTK_LIBRARIES}")
  target_include_directories(scivis PUBLIC ${VTK_INCLUDE_DIR})
  target_link_libraries(scivis PUBLIC ${VTK_LIBRARIES})
  vtk_module_autoinit(
    TARGETS scivis
    MODULES ${VTK_LIBRARIES}
  )
endif()

target_include_directories(scivis PUBLIC "$ENV{NETCDF}/include")
target_link_libraries(scivis PUBLIC $ENV{NETCDF_LIBS})

# target_link_libraries(scivis INTERFACE ${SRC_DIRS})
# target_include_directories(scivis INTERFACE ${SRC_DIRS})

# add_library(Eigen INTERFACE)
# target_include_directories(Eigen INTERFACE "/usr/include")

# add_library(LINK_LIBS INTERFACE)
# target_link_libraries(LINK_LIBS INTERFACE Boost Eigen)

# target_link_libraries(scivis INTERFACE ${LINK_LIBS})

# -Dcimg_display=0

