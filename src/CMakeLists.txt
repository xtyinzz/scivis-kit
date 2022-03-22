set (SRC_DIRS
  .
  container
  renderer
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

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(scivis PRIVATE OpenMP::OpenMP_CXX)
endif()

# target_link_libraries(scivis INTERFACE ${SRC_DIRS})
# target_include_directories(scivis INTERFACE ${SRC_DIRS})

# add_library(Eigen INTERFACE)
# target_include_directories(Eigen INTERFACE "/usr/include")

# add_library(LINK_LIBS INTERFACE)
# target_link_libraries(LINK_LIBS INTERFACE Boost Eigen)

# target_link_libraries(scivis INTERFACE ${LINK_LIBS})

# -Dcimg_display=0

