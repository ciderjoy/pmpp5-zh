cmake_minimum_required(VERSION 3.24)

project(pmpp5_solutions_examples LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PMPP_ENABLE_CUDA_WITH_CMAKE
       "Enable CUDA through CMake instead of the Windows build script" OFF)
if(PMPP_ENABLE_CUDA_WITH_CMAKE)
  include(CheckLanguage)
  check_language(CUDA)
endif()
if(PMPP_ENABLE_CUDA_WITH_CMAKE AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
endif()

function(pmpp_target_name output source)
  file(RELATIVE_PATH relative "${CMAKE_CURRENT_SOURCE_DIR}" "${source}")
  string(REGEX REPLACE "[^A-Za-z0-9]+" "_" target "${relative}")
  string(REGEX REPLACE "_$" "" target "${target}")
  set(${output} "${target}" PARENT_SCOPE)
endfunction()

file(GLOB_RECURSE PMPP_CPP_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/ch*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/appendix*.cpp")
foreach(source IN LISTS PMPP_CPP_SOURCES)
  if(source MATCHES "_mpi\\.cpp$")
    continue()
  endif()
  pmpp_target_name(target "${source}")
  add_executable("${target}" "${source}")
  target_include_directories("${target}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME "${target}" COMMAND "${target}")
endforeach()

if(PMPP_ENABLE_CUDA_WITH_CMAKE AND CMAKE_CUDA_COMPILER)
  file(GLOB_RECURSE PMPP_CUDA_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/ch*.cu")
  foreach(source IN LISTS PMPP_CUDA_SOURCES)
    pmpp_target_name(target "${source}")
    add_executable("${target}" "${source}")
    target_include_directories("${target}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    set_target_properties("${target}" PROPERTIES
                          CUDA_SEPARABLE_COMPILATION OFF)
  endforeach()
endif()

set(PMPP_MPI_CONFIGURED OFF)
if(WIN32)
  find_path(PMPP_MSMPI_INCLUDE mpi.h
            PATHS "C:/Program Files (x86)/Microsoft SDKs/MPI/Include"
            NO_DEFAULT_PATH)
  find_library(PMPP_MSMPI_LIBRARY msmpi
               PATHS "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64"
               NO_DEFAULT_PATH)
  find_program(PMPP_MSMPIEXEC mpiexec
               PATHS "C:/Program Files/Microsoft MPI/Bin"
               NO_DEFAULT_PATH)
  if(PMPP_MSMPI_INCLUDE AND PMPP_MSMPI_LIBRARY AND PMPP_MSMPIEXEC)
    file(GLOB_RECURSE PMPP_MPI_SOURCES CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/ch*_mpi.cpp")
    foreach(source IN LISTS PMPP_MPI_SOURCES)
      pmpp_target_name(target "${source}")
      add_executable("${target}" "${source}")
      target_include_directories("${target}" PRIVATE "${PMPP_MSMPI_INCLUDE}")
      target_link_libraries("${target}" PRIVATE "${PMPP_MSMPI_LIBRARY}")
      add_test(NAME "${target}"
               COMMAND "${PMPP_MSMPIEXEC}" -n 2 "$<TARGET_FILE:${target}>")
    endforeach()
    set(PMPP_MPI_CONFIGURED ON)
  endif()
endif()

if(NOT PMPP_MPI_CONFIGURED)
  find_package(MPI QUIET COMPONENTS CXX)
endif()
if(NOT PMPP_MPI_CONFIGURED AND MPI_CXX_FOUND)
  file(GLOB_RECURSE PMPP_MPI_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/ch*_mpi.cpp")
  foreach(source IN LISTS PMPP_MPI_SOURCES)
    pmpp_target_name(target "${source}")
    add_executable("${target}" "${source}")
    target_link_libraries("${target}" PRIVATE MPI::MPI_CXX)
    if(MPIEXEC_EXECUTABLE)
      add_test(NAME "${target}"
               COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
                       "$<TARGET_FILE:${target}>")
    endif()
  endforeach()
endif()
