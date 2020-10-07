cmake_minimum_required(VERSION 2.6.0)

project(NvEnc)

find_package(CUDA 7.5 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_SDK_INCLUDES})
include_directories(inc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -msse -msse2 -msse3 -msse4.1 -mavx -mavx2 -O3 -march=native -std=c++11")

file(GLOB srcs src/*.cpp)

string(TOLOWER ${PROJECT_NAME} LIBRARY_NAME)

add_library(${LIBRARY_NAME} STATIC ${srcs})
target_link_libraries(${LIBRARY_NAME} ${CUDA_LIBRARIES})

list(APPEND LIB_INC_DIR "${CMAKE_SOURCE_DIR}/inc")

# This relative path allows installed files to be relocatable.
set( CMAKECONFIG_INSTALL_DIR lib/cmake/${PROJECT_NAME} )
file( RELATIVE_PATH REL_INCLUDE_DIR
    "${CMAKE_INSTALL_PREFIX}/${CMAKECONFIG_INSTALL_DIR}"
    "${CMAKE_INSTALL_PREFIX}/include"
)

# Export library for easy inclusion from other cmake projects. APPEND allows
# call to function even as subdirectory of larger project.
FILE(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")
export( TARGETS ${LIBRARY_NAME}
        APPEND FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake" )

# Version information
configure_file(${PROJECT_NAME}ConfigVersion.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" @ONLY)

# Build tree config
set( EXPORT_LIB_INC_DIR ${LIB_INC_DIR} )
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake @ONLY IMMEDIATE )

# Install tree config
set( EXPORT_LIB_INC_DIR "\${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR}" )
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake @ONLY )

# Add package to CMake package registery for use from the build tree
option( EXPORT_${PROJECT_NAME}
  "Should the ${PROJECT_NAME} package be exported for use by other software" ON )

if( EXPORT_${PROJECT_NAME} )
  export( PACKAGE ${PROJECT_NAME} )
endif()
